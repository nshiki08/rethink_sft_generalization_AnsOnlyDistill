# @title 2. Fork の clone・revision 固定・環境構築（公式依存関係を起点。Docker は使わない）
import subprocess, shutil, json, os, sys, platform, re, time


def sh(cmd, cwd=None, check=True, capture=True, env=None):
    """subprocess の薄いラッパ。失敗時はコマンドと出力を表示する"""
    r = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=capture, text=True, env=env)
    if check and r.returncode != 0:
        print("COMMAND FAILED:", cmd)
        print(r.stdout[-4000:] if r.stdout else "", r.stderr[-4000:] if r.stderr else "")
        raise RuntimeError(f"command failed ({r.returncode})")
    return (r.stdout or "").strip()


# ---- 2.1 Fork の clone と remote 設定 -------------------------------------------------------
if not os.path.exists(os.path.join(REPO_DIR, ".git")):
    print("cloning", FORK_REPO_URL, "->", REPO_DIR)
    sh(["git", "clone", FORK_REPO_URL, REPO_DIR])
else:
    print("existing clone found at", REPO_DIR)
_origin = sh(["git", "remote", "get-url", "origin"], cwd=REPO_DIR)
assert _origin.rstrip("/").removesuffix(".git") == FORK_REPO_URL.rstrip("/").removesuffix(".git"), f"origin が Fork ではない: {_origin}"

_remotes = sh(["git", "remote"], cwd=REPO_DIR).split()
if "upstream" not in _remotes:
    sh(["git", "remote", "add", "upstream", UPSTREAM_REPO_URL], cwd=REPO_DIR)
sh(["git", "remote", "set-url", "--push", "upstream", "DISABLED_no_push_to_upstream"], cwd=REPO_DIR)   # オリジナルへの push を無効化
print("remotes:")
print(sh(["git", "remote", "-v"], cwd=REPO_DIR))
assert "DISABLED_no_push_to_upstream" in sh(["git", "remote", "get-url", "--push", "upstream"], cwd=REPO_DIR)

sh(["git", "fetch", "--quiet", "origin"], cwd=REPO_DIR)
try:
    sh(["git", "fetch", "--quiet", "upstream"], cwd=REPO_DIR)
    UPSTREAM_FETCHED = True
except RuntimeError:
    UPSTREAM_FETCHED = False
    print("WARN: upstream の fetch に失敗（参照 commit との差分確認は Fork 内の履歴で行う）")

# FORK_REF: origin のブランチ名なら origin/<ref>、そうでなければ commit として checkout（detached）
try:
    _target = sh(["git", "rev-parse", "--verify", f"origin/{FORK_REF}^{{commit}}"], cwd=REPO_DIR)
except RuntimeError:
    _target = sh(["git", "rev-parse", "--verify", f"{FORK_REF}^{{commit}}"], cwd=REPO_DIR)
sh(["git", "checkout", "--quiet", "--detach", _target], cwd=REPO_DIR)
FORK_COMMIT = sh(["git", "rev-parse", "HEAD"], cwd=REPO_DIR)
print("Fork HEAD:", FORK_COMMIT, "(ref:", FORK_REF + ")")
print("Upstream reference commit:", UPSTREAM_REFERENCE_COMMIT)

# 公式実装との差分（verl/ と training_scripts/ に変更が無いことを確認する）
try:
    _is_anc = subprocess.run(["git", "merge-base", "--is-ancestor", UPSTREAM_REFERENCE_COMMIT, "HEAD"], cwd=REPO_DIR).returncode == 0
    OFFICIAL_DIFF_STAT = sh(["git", "diff", "--stat", UPSTREAM_REFERENCE_COMMIT, "HEAD", "--", "verl", "training_scripts"], cwd=REPO_DIR)
    print("参照 commit は HEAD の祖先か:", _is_anc)
    print("公式コード (verl/, training_scripts/) と参照 commit の差分:", "なし" if not OFFICIAL_DIFF_STAT else "\n" + OFFICIAL_DIFF_STAT)
    OFFICIAL_CODE_UNCHANGED = (OFFICIAL_DIFF_STAT == "")
except RuntimeError:
    OFFICIAL_DIFF_STAT, OFFICIAL_CODE_UNCHANGED = "unknown (reference commit not available)", None
    print("WARN: 参照 commit がローカルに無く差分を確認できない")

# ---- 2.2 ハードウェア情報 -------------------------------------------------------------------
def gpu_info():
    if shutil.which("nvidia-smi") is None:
        return []
    out = sh(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"], check=False)
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append(dict(index=int(parts[0]), name=parts[1], vram_mb=int(float(parts[2])),
                             driver=parts[3] if len(parts) > 3 else None, compute_cap=parts[4] if len(parts) > 4 else None))
    return gpus


GPUS = gpu_info()
N_GPUS = len(GPUS)
_mem = {}
for line in open("/proc/meminfo"):
    k, v = line.split(":")[0], line.split(":")[1].strip().split()[0]
    _mem[k] = int(v)
_disk = shutil.disk_usage(WORK_DIR)
print(f"GPU x{N_GPUS}:", GPUS)
print(f"RAM total {_mem.get('MemTotal', 0) / 1e6:.1f} GB | disk free {_disk.free / 1e9:.1f} GB / total {_disk.total / 1e9:.1f} GB | python {platform.python_version()}")
if N_GPUS == 0:
    print("WARN: GPU が無い。dry_run とデータ処理のみ可能")

# ---- 2.3 依存関係のインストール（公式 requirements.txt の pin を起点） --------------------------
_pip = [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir"] + PIP_PINNED + EXTRA_PIP_PACKAGES
print("pip install ...", len(PIP_PINNED + EXTRA_PIP_PACKAGES), "packages (数分かかる)")
_r = subprocess.run(_pip, capture_output=True, text=True)
if _r.returncode != 0:
    print(_r.stdout[-3000:], _r.stderr[-3000:])
    raise RuntimeError("pip install failed")
print("pip install done")

# flash-attn: 公式 trainer は attn_implementation='flash_attention_2' 固定。prebuilt wheel を torch/CUDA/Python/ABI から選ぶ
_probe = subprocess.run([sys.executable, "-c",
                         "import torch, json, sys; print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda, "
                         "abi=torch._C._GLIBCXX_USE_CXX11_ABI, py=f'cp{sys.version_info.major}{sys.version_info.minor}', "
                         "cuda_available=torch.cuda.is_available(), "
                         "cap=(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None))))"],
                        capture_output=True, text=True)
if _probe.returncode != 0:
    print(_probe.stderr[-3000:]); raise RuntimeError("torch probe failed")
TORCH_PROBE = json.loads(_probe.stdout.strip().splitlines()[-1])
print("torch probe:", TORCH_PROBE)
FLASH_ATTN_OK = False
if TORCH_PROBE["cuda_available"]:
    if tuple(TORCH_PROBE["cap"]) < (8, 0):
        raise SystemExit(f"GPU compute capability {TORCH_PROBE['cap']} では flash-attn 2 が動かない（Ampere 以上が必要）。"
                         "公式 trainer は flash_attention_2 固定なので、GPU を A100/L4 などに変更する（コードは変更しない）")
    _tv = ".".join(TORCH_PROBE["torch"].split("+")[0].split(".")[:2])
    _cu = "cu" + TORCH_PROBE["cuda"].split(".")[0]
    _abi = "TRUE" if TORCH_PROBE["abi"] else "FALSE"
    _whl = (f"https://github.com/Dao-AILab/flash-attention/releases/download/v{FLASH_ATTN_VERSION}/"
            f"flash_attn-{FLASH_ATTN_VERSION}+{_cu}torch{_tv}cxx11abi{_abi}-{TORCH_PROBE['py']}-{TORCH_PROBE['py']}-linux_x86_64.whl")
    print("flash-attn wheel:", _whl)
    _r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", _whl], capture_output=True, text=True)
    if _r.returncode != 0:
        print(_r.stderr[-2000:])
        print("prebuilt wheel が無い。ソースビルドを試す（時間がかかる）")
        _r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--no-build-isolation", f"flash-attn=={FLASH_ATTN_VERSION}"],
                            capture_output=True, text=True)
        if _r.returncode != 0:
            print(_r.stderr[-3000:]); raise RuntimeError("flash-attn のインストールに失敗")
    _r = subprocess.run([sys.executable, "-c", "import flash_attn, flash_attn.bert_padding; print(flash_attn.__version__)"], capture_output=True, text=True)
    FLASH_ATTN_OK = _r.returncode == 0
    print("flash_attn import:", "OK " + _r.stdout.strip() if FLASH_ATTN_OK else "FAILED\n" + _r.stderr[-2000:])
else:
    print("CUDA なし: flash-attn は入れない（公式 trainer は CUDA 無しでは flash_attn を import しない）")

# 公式 trainer モジュールの import 確認（学習と同じ subprocess 環境）
_env = dict(os.environ, PYTHONPATH=REPO_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""))
_chk = subprocess.run([sys.executable, "-c", (
    "import json, importlib; mods=['torch','transformers','tokenizers','jinja2','accelerate','datasets','tensordict','torchdata','peft','hydra','omegaconf','wandb','ray','codetiming','numpy','pandas','pyarrow','math_verify','huggingface_hub','triton'];\n"
    "vers={}\nfor m in mods:\n    try: vers[m]=getattr(importlib.import_module(m), '__version__', 'n/a')\n    except ImportError: vers[m]='not installed'\n"
    "import verl.trainer.fsdp_sft_trainer_ours as t\n"
    "import verl.model_merger.fsdp_model_merger\n"
    "print('IMPORT_OK ' + json.dumps(vers))")], capture_output=True, text=True, env=_env, cwd=REPO_DIR)
_ok_lines = [l for l in _chk.stdout.splitlines() if l.startswith("IMPORT_OK")]
if _chk.returncode != 0 or not _ok_lines:
    print(_chk.stdout[-2000:], _chk.stderr[-4000:])
    raise RuntimeError("公式 trainer の import に失敗。不足パッケージを EXTRA_PIP_PACKAGES に追加して再実行")
PKG_VERSIONS = json.loads(_ok_lines[-1][len("IMPORT_OK "):])
import huggingface_hub as _hub_in_kernel
PKG_VERSIONS["huggingface_hub(kernel)"] = _hub_in_kernel.__version__   # カーネル側は cell 0 で import 済みなので実際の版を記録する
if PKG_VERSIONS["huggingface_hub(kernel)"] != PKG_VERSIONS.get("huggingface_hub"):
    print(f"WARN: カーネルの huggingface_hub {PKG_VERSIONS['huggingface_hub(kernel)']} と学習 subprocess の {PKG_VERSIONS.get('huggingface_hub')} が異なる。ランタイム再起動後に cell 0 から実行する")
print("import OK:", PKG_VERSIONS)

# ---- 2.4 環境記録 --------------------------------------------------------------------------
ENV_RECORD = dict(
    session_started_at=SESSION_STARTED_AT, in_colab=IN_COLAB, python=platform.python_version(), platform=platform.platform(),
    gpus=GPUS, n_gpus=N_GPUS, ram_total_gb=round(_mem.get("MemTotal", 0) / 1e6, 1), disk_free_gb=round(_disk.free / 1e9, 1),
    torch_probe=TORCH_PROBE, flash_attn_ok=FLASH_ATTN_OK, flash_attn_version=FLASH_ATTN_VERSION, packages=PKG_VERSIONS,
    fork_repo=FORK_REPO_URL, fork_ref=FORK_REF, fork_commit=FORK_COMMIT, upstream_repo=UPSTREAM_REPO_URL,
    upstream_reference_commit=UPSTREAM_REFERENCE_COMMIT, official_code_unchanged=OFFICIAL_CODE_UNCHANGED, official_diff_stat=OFFICIAL_DIFF_STAT,
)
with open(f"{RECORD_DIR}/env_record.json", "w") as f:
    json.dump(ENV_RECORD, f, indent=2, ensure_ascii=False)
print("saved", f"{RECORD_DIR}/env_record.json")
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
