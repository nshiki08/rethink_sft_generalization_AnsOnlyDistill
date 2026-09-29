# @title 1. 環境構築（Colab のカーネルで実行）: Fork の clone と公式環境（Python 3.12 + 公式 pin）の準備
# Colab の Python（2026-08 から 3.13）には公式の pin（numpy 1.26.4, ray 2.43.0 など。cp312 まで）が入らない。
# そこで公式の pin を別の Python 3.12 環境（uv venv）に入れ、その Python で 2 つ目の Jupyter カーネルを起動する。
# 設定セル以降の %%ao で始まるセルはそのカーネルで実行される（出力・エラー・停止ボタンは中継する）。
import os, sys, json, subprocess, shutil, hashlib, platform, time, signal, queue

# =============================================================================
# 環境の設定（通常は変更しない）
# =============================================================================
FORK_REPO_URL = "https://github.com/nshiki08/rethink_sft_generalization_AnsOnlyDistill"
FORK_REF = "main"                 # ブランチ名または commit。実行時に HEAD の commit を記録する
UPSTREAM_REPO_URL = "https://github.com/Nebularaid2000/rethink_sft_generalization"   # 参照専用（push 無効化）
UPSTREAM_REFERENCE_COMMIT = "71a442ea8f0adc4a1df4529d3c43393ac6e504fd"               # 再現用の参照 commit

TRAIN_PYTHON_VERSION = "3.12"     # 公式 pin の wheel がある最新版（numpy 1.26.4 / ray 2.43.0 は cp312 まで）
UV_VERSION = "0.8.17"

# 公式 requirements.txt の pin から、学習に必要な範囲を抜き出したもの（vllm/sglang/評価系は含めない）。torch 系は GPU_PROFILES で決める
PIP_PINNED = [
    "transformers==4.52.4", "tokenizers==0.21.4", "accelerate==1.10.1", "datasets==4.0.0",
    "tensordict==0.9.1", "torchdata==0.11.0", "peft==0.17.1",
    "hydra-core==1.3.2", "omegaconf==2.3.0", "wandb==0.21.1", "ray[default]==2.43.0",
    "codetiming==1.4.0", "dill==0.3.8", "pyarrow==21.0.0", "numpy==1.26.4",
    "math-verify==0.7.0", "latex2sympy2_extended==1.10.1", "pylatexenc==2.10",
    "sympy==1.13.1", "antlr4-python3-runtime==4.9.3",   # math-verify の判定結果（AO 抽出の照合）に影響するので公式と揃える
    "huggingface_hub==0.34.4", "hf-xet==1.1.9", "safetensors==0.6.2", "einops==0.8.1", "sentencepiece==0.2.1",
    "regex==2025.7.34", "word2number==1.1",   # evaluation/math_eval/utils/parser.py（既存の評価 parser）が使う
    "Jinja2==3.1.6",                          # chat template の描画に使われ、プロンプトの token に影響し得るので公式と揃える
    "pandas==2.3.3",                          # 公式 requirements.txt に pin が無い（setup.py は無指定）。導入時期で 3.x に変わらないよう 2.x の最終版に固定する
]
# 直接の pin に加え、公式 requirements.txt（clone した Fork にある）を制約 -c として使い、間接的な依存も公式と同じ版にそろえる。
# requirements.txt 自体に矛盾する pin が 2 つある（protobuf 5.29.5 と opentelemetry-proto の <5、typer 0.9.4 と fastapi-cli の >=0.15.1）ので外す
CONSTRAINT_EXCLUDE = ["protobuf", "typer"]
KERNEL_PACKAGES = ["ipykernel==6.29.5"]   # 公式環境側でカーネルを動かすためだけに使う（学習には関与しない）
EXTRA_PIP_PACKAGES = []                   # import 確認で不足が出た場合に追加

_FA = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1"
FLASH_ATTN_VERSION = "2.7.4.post1"        # 公式 pin。両プロファイルで同じ版
GPU_PROFILES = {
    # A100 / L4 / H100 など（compute capability 8.x, 9.x）: 公式 requirements.txt のまま
    "official": dict(torch=["torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0"], index_url=None, pin_overrides={},
                     flash_attn_wheel=f"{_FA}+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl", cxx11abi=False, nccl=[2, 21, 5], constraint_exclude=[],
                     deviation=None),
    # G4（RTX PRO 6000 Blackwell, sm_120）など compute capability 10 以上: torch 2.6.0 には sm_100/sm_120 のコードが無く動かない。
    # sm_120 に対応した最初の公式 torch は 2.7.0（cu128）。flash-attn は同じ 2.7.4.post1 の torch2.7 用 wheel（sm_120 を含む）。
    # torch 2.7.1 は sympy>=1.13.3 を要求するので、公式の sympy==1.13.1 と両立しない（1.13.3 に上げる）。
    "blackwell": dict(torch=["torch==2.7.1+cu128", "torchvision==0.22.1+cu128", "torchaudio==2.7.1+cu128"],
                      index_url="https://download.pytorch.org/whl/cu128", pin_overrides={"sympy": "sympy==1.13.3"},
                      constraint_exclude=["torch", "torchvision", "torchaudio", "triton", "sympy", "xformers", "vllm", "nvidia-*"],
                      flash_attn_wheel=f"{_FA}+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl", cxx11abi=True, nccl=[2, 26, 2],
                      deviation="torch 2.6.0 → 2.7.1+cu128（torchvision/torchaudio も対応版）、flash-attn は同版の torch2.7 用 wheel、"
                                "sympy 1.13.1 → 1.13.3（torch 2.7.1 の要求。math-verify の照合結果はセクション 3 のゲートで確認）。"
                                "勾配の GPU 間平均・累積・clip・state_dict 読み込みの実装は 2.6 と同じことをソースで確認済み。"
                                "NCCL 2.21.5→2.26.2, cuBLAS 12.4→12.8 などカーネル実装は変わるので丸め誤差の範囲の差が出る"),
}
ALLOW_BLACKWELL_TORCH_DEVIATION = False   # G4 を使うときだけ True にする（公式 torch からの変更として記録される）

# 論文の数学評価（evaluation/math_eval/math_eval_budget.py, vLLM）用の環境。セクション 9 で必要になったときに作る（学習用の公式環境とは別）。
# vLLM 0.8.5 は公式 requirements.txt の pin（torch==2.6.0 を要求）。依存が多いので学習用の環境には入れない。
# 公式 requirements.txt と同じ版: 下記すべて（matplotlib と pandas は requirements.txt に無い。評価スクリプトが import するので追加）
EVAL_PIP_PINNED = [
    "vllm==0.8.5", "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0", "xformers==0.0.29.post2", "triton==3.2.0",
    "transformers==4.52.4", "tokenizers==0.21.4", "numpy==1.26.4", "datasets==4.0.0", "pyarrow==21.0.0",
    "math-verify==0.7.0", "latex2sympy2_extended==1.10.1", "sympy==1.13.1", "antlr4-python3-runtime==4.9.3", "word2number==1.1",
    "regex==2025.7.34", "compressed-tensors==0.9.3", "xgrammar==0.1.18", "outlines==0.1.11", "huggingface_hub==0.34.4", "hf-xet==1.1.9",
    "pandas==2.3.3", "matplotlib==3.10.3",
]

IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
WORK_DIR = os.environ.get("AO_WORK_DIR") or ("/content/ao_work" if IN_COLAB else os.path.abspath("./ao_work"))
REPO_DIR = f"{WORK_DIR}/repo"
TRAIN_ENV_DIR = f"{WORK_DIR}/train_env"   # 公式環境（Python 3.12 venv）
RECORD_DIR = f"{WORK_DIR}/records"
for _d in (WORK_DIR, RECORD_DIR):
    os.makedirs(_d, exist_ok=True)


def sh(cmd, cwd=None, check=True, env=None):
    """subprocess の薄いラッパ。失敗時はコマンドと出力の末尾を表示する"""
    r = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=True, text=True, env=env)
    if check and r.returncode != 0:
        print("COMMAND FAILED:", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)))
        print((r.stdout or "")[-4000:], (r.stderr or "")[-4000:])
        raise RuntimeError(f"command failed ({r.returncode})")
    return (r.stdout or "").strip()


# ---- 1.1 GPU の確認とプロファイルの決定 -----------------------------------------------------------
def gpu_info():
    if shutil.which("nvidia-smi") is None:
        return []
    out = sh(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"], check=False)
    gpus = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5:
            gpus.append(dict(index=int(p[0]), name=p[1], vram_mb=int(float(p[2])), driver=p[3], compute_cap=p[4]))
    return gpus


GPUS = gpu_info()
print(f"GPU x{len(GPUS)}:", GPUS or "なし（dry_run とデータ処理のみ可能）")
_caps = sorted({tuple(int(v) for v in g["compute_cap"].split(".")) for g in GPUS})
if len(_caps) > 1:
    raise SystemExit(f"種類の違う GPU が混在している: {_caps}")
GPU_CAP = _caps[0] if _caps else None
if GPU_CAP is not None and GPU_CAP < (8, 0):
    raise SystemExit(f"GPU compute capability {GPU_CAP} では flash-attn 2 が動かない（Ampere 以上が必要）。"
                     "公式 trainer は flash_attention_2 固定なので、ランタイムを A100 などに変更する（コードは変更しない）")
if GPU_CAP is not None and GPU_CAP >= (10, 0):
    if not ALLOW_BLACKWELL_TORCH_DEVIATION:
        raise SystemExit(
            f"GPU {GPUS[0]['name']}（compute capability {GPU_CAP[0]}.{GPU_CAP[1]}）では公式の torch 2.6.0 が動かない（sm_100/sm_120 のコードが無い）。\n"
            "  - 公式環境のまま学習する: ランタイムを A100 に変更する（推奨）\n"
            "  - このまま使う: ALLOW_BLACKWELL_TORCH_DEVIATION=True にして再実行する。torch 2.7.1+cu128 を使い、公式との差として記録する\n"
            f"    変更内容: {GPU_PROFILES['blackwell']['deviation']}")
    GPU_PROFILE = "blackwell"
else:
    GPU_PROFILE = "official"
PROFILE = GPU_PROFILES[GPU_PROFILE]
print("GPU プロファイル:", GPU_PROFILE, "| 公式からの変更:", PROFILE["deviation"] or "なし")

# ---- 1.2 Fork の clone と remote 設定 -------------------------------------------------------------
if not os.path.exists(os.path.join(REPO_DIR, ".git")):
    print("cloning", FORK_REPO_URL, "->", REPO_DIR)
    sh(["git", "clone", "--quiet", FORK_REPO_URL, REPO_DIR])
_origin = sh(["git", "remote", "get-url", "origin"], cwd=REPO_DIR)
assert _origin.rstrip("/").removesuffix(".git") == FORK_REPO_URL.rstrip("/").removesuffix(".git"), f"origin が Fork ではない: {_origin}"
if "upstream" not in sh(["git", "remote"], cwd=REPO_DIR).split():
    sh(["git", "remote", "add", "upstream", UPSTREAM_REPO_URL], cwd=REPO_DIR)
sh(["git", "remote", "set-url", "--push", "upstream", "DISABLED_no_push_to_upstream"], cwd=REPO_DIR)   # オリジナルへの push を無効化
assert "DISABLED_no_push_to_upstream" in sh(["git", "remote", "get-url", "--push", "upstream"], cwd=REPO_DIR)
print(sh(["git", "remote", "-v"], cwd=REPO_DIR))
sh(["git", "fetch", "--quiet", "origin"], cwd=REPO_DIR)
try:
    sh(["git", "fetch", "--quiet", "upstream"], cwd=REPO_DIR)
    UPSTREAM_FETCHED = True
except RuntimeError:
    UPSTREAM_FETCHED = False
    print("WARN: upstream の fetch に失敗（参照 commit との差分確認は Fork 内の履歴で行う）")
try:
    _target = sh(["git", "rev-parse", "--verify", f"origin/{FORK_REF}^{{commit}}"], cwd=REPO_DIR)
except RuntimeError:
    _target = sh(["git", "rev-parse", "--verify", f"{FORK_REF}^{{commit}}"], cwd=REPO_DIR)
sh(["git", "checkout", "--quiet", "--detach", _target], cwd=REPO_DIR)
FORK_COMMIT = sh(["git", "rev-parse", "HEAD"], cwd=REPO_DIR)
print("Fork HEAD:", FORK_COMMIT, "(ref:", FORK_REF + ")")
try:
    _is_anc = subprocess.run(["git", "merge-base", "--is-ancestor", UPSTREAM_REFERENCE_COMMIT, "HEAD"], cwd=REPO_DIR).returncode == 0
    OFFICIAL_DIFF_STAT = sh(["git", "diff", "--stat", UPSTREAM_REFERENCE_COMMIT, "HEAD", "--", "verl", "training_scripts"], cwd=REPO_DIR)
    OFFICIAL_CODE_UNCHANGED = (OFFICIAL_DIFF_STAT == "")
    print("参照 commit は HEAD の祖先か:", _is_anc, "| 公式コード (verl/, training_scripts/) の差分:", "なし" if OFFICIAL_CODE_UNCHANGED else "\n" + OFFICIAL_DIFF_STAT)
except RuntimeError:
    OFFICIAL_DIFF_STAT, OFFICIAL_CODE_UNCHANGED = "unknown (reference commit not available)", None
    print("WARN: 参照 commit がローカルに無く差分を確認できない")

# ---- 1.3 前回の公式環境カーネルと学習プロセスの終了 ----------------------------------------------------
# Colab の「セッションを再起動」やセル 1 の再実行では、前回の公式環境カーネル（GPU メモリを持っている可能性）と、
# そこから別の process group で起動した学習プロセス（torchrun）が残る。pid ファイルに記録した process group を、
# cmdline が期待どおりのプロセスだけ終了する（pid の再利用で無関係なプロセスを止めない。自分自身は止めない）。
def _group_members(pgid):
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            stat = open(f"/proc/{d}/stat").read()
            if int(stat.rsplit(")", 1)[1].split()[2]) == pgid:
                out.append((int(d), open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")))
        except (OSError, IndexError, ValueError):
            pass
    return out


def kill_recorded_group(pidfile, must_contain, label):
    try:
        pgid = int(open(pidfile).read().strip())
    except (OSError, ValueError):
        return
    members = [(p, c) for p, c in _group_members(pgid) if p != os.getpid()]
    if pgid == os.getpgid(0) or not members or not any(all(m in c for m in must_contain) for _, c in members):
        return   # 既に終了している、または pid が別のプロセスに再利用されている
    try:
        os.killpg(pgid, signal.SIGKILL)
        print(f"前回の{label}（process group {pgid}, {len(members)} プロセス）を終了した")
    except ProcessLookupError:
        pass


TRAIN_PY = f"{TRAIN_ENV_DIR}/bin/python"
kill_recorded_group(f"{WORK_DIR}/ao-trainer.pgid", ["torch.distributed.run"], "学習プロセス")
kill_recorded_group(f"{WORK_DIR}/ao-train-env.pid", ["ipykernel_launcher", TRAIN_ENV_DIR], "公式環境カーネル")

# ---- 1.4 公式環境（Python 3.12 venv）の作成と公式 pin の導入 ------------------------------------------
# Colab は uv 用の制約ファイルを UV_* 環境変数で指定していることがある（Colab 自身の numpy 2.x などに固定される）。公式環境には使わない
_uv_env = {k: v for k, v in os.environ.items() if not k.startswith("UV_")}
_uv_env["UV_CACHE_DIR"] = os.environ.get("AO_UV_CACHE_DIR", f"{WORK_DIR}/uv_cache")
_uv_env["UV_PYTHON_INSTALL_DIR"] = f"{WORK_DIR}/uv_python"
# UV_LINK_MODE は既定（Linux では hardlink）。cache と venv が同じディスクなので CUDA 系 wheel を二重に置かない
try:
    sh([sys.executable, "-m", "uv", "--version"])
except RuntimeError:
    sh([sys.executable, "-m", "pip", "install", "-q", f"uv=={UV_VERSION}"])
UV = [sys.executable, "-m", "uv", "--no-config"]
_pins = [PROFILE["pin_overrides"].get(p.split("==")[0].split("[")[0].lower(), p) for p in PIP_PINNED + EXTRA_PIP_PACKAGES]

def write_constraints(path, exclude):
    """公式 requirements.txt の == 行を制約ファイルにする（exclude は小文字・ハイフン区切りの名前か glob）"""
    import fnmatch
    keep, dropped = [], []
    for line in open(f"{REPO_DIR}/requirements.txt"):
        line = line.split("#")[0].strip()
        if "==" not in line:
            continue
        name = line.split("==")[0].strip().lower().replace("_", "-")
        (dropped if any(fnmatch.fnmatch(name, pat) for pat in exclude) else keep).append(line)
    with open(path, "w") as f:
        f.write("\n".join(keep) + "\n")
    return keep, dropped


TRAIN_CONSTRAINTS = f"{WORK_DIR}/constraints_train_{GPU_PROFILE}.txt"
_cons_keep, _cons_dropped = write_constraints(TRAIN_CONSTRAINTS, CONSTRAINT_EXCLUDE + PROFILE["constraint_exclude"])
EVAL_CONSTRAINTS = f"{WORK_DIR}/constraints_eval.txt"          # 評価用環境（vLLM 0.8.5、official プロファイルのみ）
write_constraints(EVAL_CONSTRAINTS, CONSTRAINT_EXCLUDE)
print(f"公式 requirements.txt を制約に使う: {len(_cons_keep)} 件（除外 {len(_cons_dropped)} 件: {sorted(set(l.split('==')[0] for l in _cons_dropped))}）")
_torch_index = os.environ.get("AO_TORCH_INDEX_URL") or PROFILE["index_url"]   # AO_TORCH_INDEX_URL はローカル（CPU）検証用
_want = dict(python=TRAIN_PYTHON_VERSION, torch=PROFILE["torch"], index=_torch_index, pins=_pins + KERNEL_PACKAGES,
             flash_attn=PROFILE["flash_attn_wheel"] if GPU_CAP else None, constraints=_cons_keep)
_want_hash = hashlib.sha256(json.dumps(_want, sort_keys=True).encode()).hexdigest()[:16]
_marker = f"{TRAIN_ENV_DIR}/ao_env_ok.json"
if os.path.isfile(_marker) and json.load(open(_marker)).get("hash") == _want_hash and os.path.isfile(TRAIN_PY):
    print("公式環境は作成済み（同じ pin）:", TRAIN_ENV_DIR)
else:
    t0 = time.time()
    shutil.rmtree(TRAIN_ENV_DIR, ignore_errors=True)
    sh(UV + ["venv", "--quiet", "--python", TRAIN_PYTHON_VERSION, "--python-preference", "only-managed", TRAIN_ENV_DIR], env=_uv_env)
    _idx = ["--index-url", _torch_index, "--extra-index-url", "https://pypi.org/simple", "--index-strategy", "unsafe-best-match"] if _torch_index else []
    print(f"公式 pin を導入中（{len(_want['pins']) + 3} packages。数分かかる）...")
    sh(UV + ["pip", "install", "--quiet", "--python", TRAIN_PY, "-c", TRAIN_CONSTRAINTS] + _idx + PROFILE["torch"] + _want["pins"], env=_uv_env)
    if GPU_CAP:
        # flash-attn: 公式 trainer は attn_implementation='flash_attention_2' 固定。依存（torch, einops）は導入済みなので --no-deps
        sh(UV + ["pip", "install", "--quiet", "--python", TRAIN_PY, "--no-deps", PROFILE["flash_attn_wheel"]], env=_uv_env)
    json.dump(dict(hash=_want_hash, want=_want, created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), open(_marker, "w"), indent=2)
    print(f"公式環境を作成した: {TRAIN_ENV_DIR}（{time.time() - t0:.0f} 秒）")
TRAIN_ENV_FREEZE = f"{RECORD_DIR}/train_env_freeze.txt"   # 公式環境の全パッケージの版（記録）
with open(TRAIN_ENV_FREEZE, "w") as f:
    f.write(sh(UV + ["pip", "freeze", "--python", TRAIN_PY], env=_uv_env) + "\n")

# ---- 1.5 公式環境の確認（Python, torch, GPU アーキテクチャ, flash-attn, 公式 trainer の import） -------------------
_TRAIN_ENV_VARS = {k: v for k, v in os.environ.items() if not k.startswith("UV_") and k not in ("PYTHONPATH", "PYTHONHOME", "MPLBACKEND", "VIRTUAL_ENV")}
_TRAIN_ENV_VARS.update(PYTHONPATH=REPO_DIR, VIRTUAL_ENV=TRAIN_ENV_DIR, PATH=f"{TRAIN_ENV_DIR}/bin" + os.pathsep + os.environ.get("PATH", ""),
                       AO_WORK_DIR=WORK_DIR)
_probe_src = r'''
import json, sys, importlib, platform
mods = ["torch", "transformers", "tokenizers", "jinja2", "accelerate", "datasets", "tensordict", "torchdata", "peft", "hydra", "omegaconf",
        "wandb", "ray", "codetiming", "numpy", "pandas", "pyarrow", "math_verify", "huggingface_hub", "triton", "flash_attn", "ipykernel"]
vers = {}
for m in mods:
    try:
        vers[m] = getattr(importlib.import_module(m), "__version__", "n/a")
    except ImportError:
        vers[m] = "not installed"
import torch
p = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
         cuda_available=torch.cuda.is_available(), arch_list=torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
         cap=list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
         cudnn=torch.backends.cudnn.version() if torch.cuda.is_available() else None,
         nccl=list(torch.cuda.nccl.version()) if torch.cuda.is_available() else None)
if p["cuda_available"]:
    import flash_attn, flash_attn.bert_padding  # noqa: F401
    from flash_attn import flash_attn_func
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    p["cuda_matmul_ok"] = bool(torch.isfinite(x @ x).all())   # その GPU 用のカーネルが無ければここで失敗する
    q, k, v = (torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    o = flash_attn_func(q, k, v, causal=True)
    o.float().sum().backward()                                 # flash-attn の forward / backward カーネルを GPU で実行する
    p["flash_attn_kernel_ok"] = bool(torch.isfinite(o).all() and torch.isfinite(q.grad).all())
import verl.trainer.fsdp_sft_trainer_ours  # noqa: F401
import verl.model_merger.fsdp_model_merger  # noqa: F401
print("PROBE_OK " + json.dumps(dict(probe=p, packages=vers)))
'''
_r = subprocess.run([TRAIN_PY, "-c", _probe_src], capture_output=True, text=True, env=_TRAIN_ENV_VARS, cwd=REPO_DIR)
_ok = [l for l in _r.stdout.splitlines() if l.startswith("PROBE_OK ")]
if _r.returncode != 0 or not _ok:
    print(_r.stdout[-3000:], _r.stderr[-5000:])
    raise RuntimeError("公式環境の確認に失敗。不足パッケージは EXTRA_PIP_PACKAGES に追加して再実行する")
_probe = json.loads(_ok[-1][len("PROBE_OK "):])
TORCH_PROBE, PKG_VERSIONS = _probe["probe"], _probe["packages"]
assert TORCH_PROBE["python"].startswith(TRAIN_PYTHON_VERSION + "."), TORCH_PROBE["python"]
if GPU_CAP:
    # 同じ major の小さい minor 向けコード（例: L4 の sm_89 で sm_86/sm_80）は動く。PTX（compute_XY）は XY <= 実機なら JIT で動く
    def _arch_ok(arch_list, cap):
        for a in arch_list:
            kind, _, num = a.partition("_")
            if not num.isdigit():
                continue
            major, minor = int(num[:-1]), int(num[-1])
            if kind == "sm" and major == cap[0] and minor <= cap[1]:
                return True
            if kind == "compute" and (major, minor) <= tuple(cap):
                return True
        return False

    assert _arch_ok(TORCH_PROBE["arch_list"], GPU_CAP), f"torch {TORCH_PROBE['torch']} に compute capability {GPU_CAP} 用のコードが無い: {TORCH_PROBE['arch_list']}"
    assert TORCH_PROBE.get("cuda_matmul_ok"), "GPU 上の行列積に失敗"
    assert TORCH_PROBE.get("flash_attn_kernel_ok"), "flash-attn のカーネル実行に失敗"
    assert TORCH_PROBE["abi"] == PROFILE["cxx11abi"], f"torch の C++ ABI {TORCH_PROBE['abi']} が flash-attn wheel（cxx11abi={PROFILE['cxx11abi']}）と合わない"
    if TORCH_PROBE["nccl"] != PROFILE["nccl"]:
        print(f"WARN: NCCL {TORCH_PROBE['nccl']} が想定 {PROFILE['nccl']} と異なる（記録する）")
print("公式環境:", {k: TORCH_PROBE[k] for k in ("python", "torch", "cuda", "cap", "cudnn", "nccl")})
print("packages:", PKG_VERSIONS)

ENV_BOOTSTRAP = dict(
    kernel_python=platform.python_version(), train_python=TORCH_PROBE["python"], train_env_dir=TRAIN_ENV_DIR, train_env_hash=_want_hash,
    gpu_profile=GPU_PROFILE, gpu_profile_deviation=PROFILE["deviation"], torch_pins=PROFILE["torch"], torch_index_url=_torch_index,
    flash_attn_version=FLASH_ATTN_VERSION, flash_attn_wheel=_want["flash_attn"], pip_pinned=_pins, pin_overrides=PROFILE["pin_overrides"],
    gpus=GPUS, torch_probe=TORCH_PROBE, packages=PKG_VERSIONS, in_colab=IN_COLAB, work_dir=WORK_DIR, repo_dir=REPO_DIR,
    fork_repo=FORK_REPO_URL, fork_ref=FORK_REF, fork_commit=FORK_COMMIT, upstream_repo=UPSTREAM_REPO_URL,
    upstream_reference_commit=UPSTREAM_REFERENCE_COMMIT, upstream_fetched=UPSTREAM_FETCHED,
    official_code_unchanged=OFFICIAL_CODE_UNCHANGED, official_diff_stat=OFFICIAL_DIFF_STAT,
    # セクション 9 が評価用の環境を作るときに使う（uv は Colab のカーネルの Python に入っている）
    uv_cmd=UV, uv_cache_dir=_uv_env["UV_CACHE_DIR"], uv_python_install_dir=_uv_env["UV_PYTHON_INSTALL_DIR"],
    eval_env_dir=f"{WORK_DIR}/eval_env", eval_pip_pinned=EVAL_PIP_PINNED, eval_constraints=EVAL_CONSTRAINTS,
    train_constraints=TRAIN_CONSTRAINTS, constraint_excluded=_cons_dropped, train_env_freeze=TRAIN_ENV_FREEZE,
)
ENV_BOOTSTRAP_PATH = f"{RECORD_DIR}/env_bootstrap.json"
json.dump(ENV_BOOTSTRAP, open(ENV_BOOTSTRAP_PATH, "w"), indent=2, ensure_ascii=False)


# ---- 1.6 公式環境のカーネルを起動し、%%ao マジックを登録する --------------------------------------------
class TrainEnvKernel:
    """公式環境（別 Python）で動く Jupyter カーネル。コードを送り、出力・エラー・停止を中継する"""

    def __init__(self, python_exe, workdir, env, cwd, name="ao-train-env"):
        self.python_exe, self.workdir, self.env, self.cwd, self.name = python_exe, workdir, env, cwd, name
        self.pidfile = os.path.join(workdir, f"{name}.pid")
        self.km = self.kc = None

    def start(self):
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        spec_dir = os.path.join(self.workdir, "kernelspec", self.name)
        os.makedirs(spec_dir, exist_ok=True)
        with open(os.path.join(spec_dir, "kernel.json"), "w") as f:
            json.dump(dict(argv=[self.python_exe, "-m", "ipykernel_launcher", "-f", "{connection_file}"], display_name=self.name, language="python"), f)
        ksm = KernelSpecManager()
        ksm.kernel_dirs = [os.path.dirname(spec_dir)]
        # IPC（Unix socket）で通信する。パス長の上限（108 byte）があるので短いパスにする
        self.km = KernelManager(kernel_name=self.name, kernel_spec_manager=ksm, transport="ipc", ip=f"/tmp/aok{os.getpid()}")
        # 新しい process group（後で子プロセスごと終了できる）。カーネルの fd 直書き出力（subprocess など）は既に中継されるのでログファイルへ
        self._log = open(os.path.join(self.workdir, f"{self.name}.log"), "ab")
        self.km.start_kernel(env=dict(self.env), cwd=self.cwd, start_new_session=True, stdout=self._log, stderr=self._log)
        with open(self.pidfile, "w") as f:
            f.write(str(self.km.provisioner.pid if getattr(self.km, "provisioner", None) else self.km.kernel.pid))
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=180)
        return self

    def alive(self):
        return self.km is not None and self.km.is_alive()

    def run(self, code):
        if not self.alive():
            raise RuntimeError("公式環境カーネルが動いていない（メモリ不足で落ちた可能性）。セル 1（環境構築）を再実行する")
        msg_id = self.kc.execute(code, store_history=False, allow_stdin=False)
        error, interrupted = None, False
        while True:
            try:   # 停止ボタン（KeyboardInterrupt）がループのどこで届いても公式環境カーネルに割り込みを送り、終了を待つ
                try:
                    msg = self.kc.get_iopub_msg(timeout=1)
                except queue.Empty:
                    if not self.alive():
                        raise RuntimeError("公式環境カーネルが終了した（メモリ不足など）。セル 1（環境構築）から実行し直す")
                    continue
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                t, c = msg["msg_type"], msg["content"]
                if t == "stream":
                    (sys.stderr if c["name"] == "stderr" else sys.stdout).write(c["text"])
                elif t in ("display_data", "execute_result"):
                    try:
                        from IPython.display import publish_display_data
                        publish_display_data(c["data"], c.get("metadata", {}))
                    except Exception:
                        print(c["data"].get("text/plain", ""))
                elif t == "error":
                    error = c
                    sys.stderr.write("\n".join(c["traceback"]) + "\n")
                elif t == "status" and c["execution_state"] == "idle":
                    break
            except KeyboardInterrupt:
                interrupted = True
                self.km.interrupt_kernel()
        sys.stdout.flush(), sys.stderr.flush()
        if interrupted:
            raise KeyboardInterrupt
        if error is not None:
            raise RuntimeError(f"公式環境カーネルでエラー: {error['ename']}: {error['evalue']}")


_kernel_env = dict(_TRAIN_ENV_VARS, AO_ENV_BOOTSTRAP=ENV_BOOTSTRAP_PATH)   # HF_TOKEN / WANDB_API_KEY（セル 0 で設定）もここで引き継ぐ
AO_KERNEL = TrainEnvKernel(TRAIN_PY, WORK_DIR, _kernel_env, cwd=WORK_DIR).start()
AO_KERNEL.run("import json, os\n"
              "ENV_BOOTSTRAP = json.load(open(os.environ['AO_ENV_BOOTSTRAP']))\n"
              "import sys, platform; print('公式環境カーネル:', sys.executable, platform.python_version())")

try:
    from IPython import get_ipython
    from IPython.core.magic import register_cell_magic

    @register_cell_magic
    def ao(line, cell):
        """%%ao: このセルを公式環境のカーネルで実行する"""
        AO_KERNEL.run(cell)

    print("%%ao を登録した。設定セル以降は公式環境のカーネルで実行される")
except Exception as _e:   # IPython の外（スクリプト実行）では登録しない
    print("WARN: %%ao を登録できない:", _e)
