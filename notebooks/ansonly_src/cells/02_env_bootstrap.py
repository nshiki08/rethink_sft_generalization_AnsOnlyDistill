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
KERNEL_PACKAGES = ["ipykernel==6.29.5"]   # 公式環境側でカーネルを動かすためだけに使う（学習には関与しない）
EXTRA_PIP_PACKAGES = []                   # import 確認で不足が出た場合に追加

_FA = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1"
FLASH_ATTN_VERSION = "2.7.4.post1"        # 公式 pin。両プロファイルで同じ版
GPU_PROFILES = {
    # A100 / L4 / H100 など（compute capability 8.x, 9.x）: 公式 requirements.txt のまま
    "official": dict(torch=["torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0"], index_url=None,
                     flash_attn_wheel=f"{_FA}+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl", deviation=None),
    # G4（RTX PRO 6000 Blackwell, sm_120）など compute capability 10 以上: torch 2.6.0 には sm_100/sm_120 のコードが無く動かない。
    # sm_120 に対応した最初の公式 torch は 2.7.0（cu128）。flash-attn は同じ 2.7.4.post1 の torch2.7 用 wheel（sm_120 を含む）。
    "blackwell": dict(torch=["torch==2.7.1+cu128", "torchvision==0.22.1+cu128", "torchaudio==2.7.1+cu128"],
                      index_url="https://download.pytorch.org/whl/cu128",
                      flash_attn_wheel=f"{_FA}+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
                      deviation="torch 2.6.0 → 2.7.1+cu128（torchvision/torchaudio も対応版）、flash-attn は同版の torch2.7 用 wheel。"
                                "勾配の GPU 間平均・累積・clip・state_dict 読み込みの実装は 2.6 と同じことをソースで確認済み。"
                                "NCCL 2.21.5→2.26.2, cuBLAS 12.4→12.8 などカーネル実装は変わるので丸め誤差の範囲の差が出る"),
}
ALLOW_BLACKWELL_TORCH_DEVIATION = False   # G4 を使うときだけ True にする（公式 torch からの変更として記録される）

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

# ---- 1.3 公式環境（Python 3.12 venv）の作成と公式 pin の導入 ------------------------------------------
# Colab は uv 用の制約ファイルを UV_* 環境変数で指定していることがある（Colab 自身の numpy 2.x などに固定される）。公式環境には使わない
_uv_env = {k: v for k, v in os.environ.items() if not k.startswith("UV_")}
_uv_env["UV_CACHE_DIR"] = os.environ.get("AO_UV_CACHE_DIR", f"{WORK_DIR}/uv_cache")
_uv_env["UV_PYTHON_INSTALL_DIR"] = f"{WORK_DIR}/uv_python"
_uv_env["UV_LINK_MODE"] = "copy"
try:
    sh([sys.executable, "-m", "uv", "--version"])
except RuntimeError:
    sh([sys.executable, "-m", "pip", "install", "-q", f"uv=={UV_VERSION}"])
UV = [sys.executable, "-m", "uv", "--no-config"]
TRAIN_PY = f"{TRAIN_ENV_DIR}/bin/python"

_torch_index = os.environ.get("AO_TORCH_INDEX_URL") or PROFILE["index_url"]   # AO_TORCH_INDEX_URL はローカル（CPU）検証用
_want = dict(python=TRAIN_PYTHON_VERSION, torch=PROFILE["torch"], index=_torch_index, pins=PIP_PINNED + EXTRA_PIP_PACKAGES + KERNEL_PACKAGES,
             flash_attn=PROFILE["flash_attn_wheel"] if GPU_CAP else None)
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
    sh(UV + ["pip", "install", "--quiet", "--python", TRAIN_PY] + _idx + PROFILE["torch"] + _want["pins"], env=_uv_env)
    if GPU_CAP:
        # flash-attn: 公式 trainer は attn_implementation='flash_attention_2' 固定。依存（torch, einops）は導入済みなので --no-deps
        sh(UV + ["pip", "install", "--quiet", "--python", TRAIN_PY, "--no-deps", PROFILE["flash_attn_wheel"]], env=_uv_env)
    json.dump(dict(hash=_want_hash, want=_want, created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), open(_marker, "w"), indent=2)
    print(f"公式環境を作成した: {TRAIN_ENV_DIR}（{time.time() - t0:.0f} 秒）")

# ---- 1.4 公式環境の確認（Python, torch, GPU アーキテクチャ, flash-attn, 公式 trainer の import） -------------------
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
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    p["cuda_matmul_ok"] = bool(torch.isfinite(x @ x).all())   # その GPU 用のカーネルが無ければここで失敗する
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
    _sm = f"sm_{GPU_CAP[0]}{GPU_CAP[1]}"
    assert _sm in TORCH_PROBE["arch_list"] or any(a == f"compute_{GPU_CAP[0]}{GPU_CAP[1]}" for a in TORCH_PROBE["arch_list"]), \
        f"torch {TORCH_PROBE['torch']} に {_sm} のコードが無い: {TORCH_PROBE['arch_list']}"
    assert TORCH_PROBE.get("cuda_matmul_ok"), "GPU 上の行列積に失敗"
print("公式環境:", {k: TORCH_PROBE[k] for k in ("python", "torch", "cuda", "cap", "cudnn", "nccl")})
print("packages:", PKG_VERSIONS)

ENV_BOOTSTRAP = dict(
    kernel_python=platform.python_version(), train_python=TORCH_PROBE["python"], train_env_dir=TRAIN_ENV_DIR, train_env_hash=_want_hash,
    gpu_profile=GPU_PROFILE, gpu_profile_deviation=PROFILE["deviation"], torch_pins=PROFILE["torch"], torch_index_url=_torch_index,
    flash_attn_version=FLASH_ATTN_VERSION, flash_attn_wheel=_want["flash_attn"], pip_pinned=PIP_PINNED + EXTRA_PIP_PACKAGES,
    gpus=GPUS, torch_probe=TORCH_PROBE, packages=PKG_VERSIONS, in_colab=IN_COLAB, work_dir=WORK_DIR, repo_dir=REPO_DIR,
    fork_repo=FORK_REPO_URL, fork_ref=FORK_REF, fork_commit=FORK_COMMIT, upstream_repo=UPSTREAM_REPO_URL,
    upstream_reference_commit=UPSTREAM_REFERENCE_COMMIT, upstream_fetched=UPSTREAM_FETCHED,
    official_code_unchanged=OFFICIAL_CODE_UNCHANGED, official_diff_stat=OFFICIAL_DIFF_STAT,
)
ENV_BOOTSTRAP_PATH = f"{RECORD_DIR}/env_bootstrap.json"
json.dump(ENV_BOOTSTRAP, open(ENV_BOOTSTRAP_PATH, "w"), indent=2, ensure_ascii=False)


# ---- 1.5 公式環境のカーネルを起動し、%%ao マジックを登録する --------------------------------------------
class TrainEnvKernel:
    """公式環境（別 Python）で動く Jupyter カーネル。コードを送り、出力・エラー・停止を中継する"""

    def __init__(self, python_exe, workdir, env, cwd, name="ao-train-env"):
        self.python_exe, self.workdir, self.env, self.cwd, self.name = python_exe, workdir, env, cwd, name
        self.pidfile = os.path.join(workdir, f"{name}.pid")
        self.km = self.kc = None

    def _kill_previous(self):
        # Colab のセッション再起動などで取り残された前回のカーネル（GPU メモリを持ったままの可能性）と、その子プロセスを終了する
        try:
            pid = int(open(self.pidfile).read().strip())
            os.kill(pid, 0)
        except Exception:
            return
        try:
            os.killpg(pid, signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        print(f"前回の公式環境カーネル (pid {pid}) を終了した")

    def start(self):
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        self._kill_previous()
        spec_dir = os.path.join(self.workdir, "kernelspec", self.name)
        os.makedirs(spec_dir, exist_ok=True)
        with open(os.path.join(spec_dir, "kernel.json"), "w") as f:
            json.dump(dict(argv=[self.python_exe, "-m", "ipykernel_launcher", "-f", "{connection_file}"], display_name=self.name, language="python"), f)
        ksm = KernelSpecManager()
        ksm.kernel_dirs = [os.path.dirname(spec_dir)]
        # IPC（Unix socket）で通信する。パス長の上限（108 byte）があるので短いパスにする
        self.km = KernelManager(kernel_name=self.name, kernel_spec_manager=ksm, transport="ipc", ip=f"/tmp/aok{os.getpid()}")
        self.km.start_kernel(env=dict(self.env), cwd=self.cwd, start_new_session=True)   # 新しい process group（後で子プロセスごと終了できる）
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
            try:
                msg = self.kc.get_iopub_msg(timeout=1)
            except queue.Empty:
                if not self.alive():
                    raise RuntimeError("公式環境カーネルが終了した（メモリ不足など）。セル 1（環境構築）から実行し直す")
                continue
            except KeyboardInterrupt:
                interrupted = True            # 停止ボタン → 公式環境カーネルにも割り込みを送り、終了を待つ
                self.km.interrupt_kernel()
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
