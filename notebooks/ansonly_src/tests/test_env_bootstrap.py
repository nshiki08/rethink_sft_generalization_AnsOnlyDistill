"""End-to-end CPU test of the Colab flow: cell 0 and cell 1 in a "Colab kernel" Python, then cells 03.. in the official-env kernel.

Run this with a Python that mimics the Colab kernel (e.g. Python 3.13 + jupyter_client 7.4.9 + ipython 7.34, and `uv` importable
as a module), NOT with the official-pin Python:
    AO_TEST_WORK_DIR=... <colab-like python> notebooks/ansonly_src/tests/test_env_bootstrap.py [up_to_cell_prefix]
    AO_TEST_WORK_DIR=... <colab-like ipython> notebooks/ansonly_src/tests/test_env_bootstrap.py [up_to]   # %%ao マジック経由
Cell 1 clones the Fork from GitHub, creates the Python 3.12 venv with uv and installs the official pins
(CPU torch from $AO_TORCH_INDEX_URL, default https://download.pytorch.org/whl/cpu), then starts the official-env kernel.
Cells 03.. are sent to that kernel exactly as %%ao does. The base model is replaced by its tokenizer/config (no weights).
"""
import os, sys, glob, time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
TEST_WORK = os.path.expanduser(os.environ.get("AO_TEST_WORK_DIR", "~/.cache/ao_nb_test"))
os.environ["AO_WORK_DIR"] = os.path.join(TEST_WORK, "ao_work_bootstrap")
os.environ.setdefault("AO_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cpu")
up_to = sys.argv[1] if len(sys.argv) > 1 else "08"
cells = sorted(glob.glob(os.path.join(SRC, "cells", "*.py")))
g = {"__name__": "__main__"}
print("colab-like kernel python:", sys.version.split()[0], sys.executable)
try:
    from IPython import get_ipython
    IP = get_ipython()
except ImportError:
    IP = None
print("IPython shell:", type(IP).__name__ if IP else None)

STUB_BASE_MODEL = '''
def get_base_model_local():
    """テスト用: Base の tokenizer と config だけを取得する（重みは取得しない）"""
    from huggingface_hub import snapshot_download
    d = f"{MODELS_DIR}/{MODEL_KEY}-tokenizer-only"
    if not os.path.isfile(os.path.join(d, "tokenizer.json")):
        snapshot_download(MODEL_INFO["base_repo"], revision=MODEL_INFO["base_revision"], local_dir=d, allow_patterns=["*.json", "*.txt"])
    return d
print("[test stub] get_base_model_local -> tokenizer/config only")
'''

for path in cells:
    name = os.path.basename(path)
    if name[:2] > up_to:
        break
    src = open(path).read()
    t0 = time.time()
    print(f"\n{'#' * 30} RUN {name} {'#' * 30}", flush=True)
    if name in ("01_auth.py", "02_env_bootstrap.py"):
        exec(compile(src, name, "exec"), g)          # Colab のカーネルで実行するセル
    elif IP is not None:
        IP.run_cell_magic("ao", "", src)              # ipython で実行したときは %%ao マジックそのものを使う
    else:
        g["AO_KERNEL"].run(src)                       # %%ao と同じ経路
    if name.startswith("06_"):
        g["AO_KERNEL"].run(STUB_BASE_MODEL)
    print(f"{'#' * 30} DONE {name} in {time.time() - t0:.1f}s", flush=True)

k = g["AO_KERNEL"]
k.run("import sys, numpy, ray, torch, transformers\n"
      "print('KERNEL_CHECK', sys.version.split()[0], numpy.__version__, ray.__version__, torch.__version__, transformers.__version__)\n"
      "print('GATES', {k: globals().get(k) for k in ('AO_DATA_READY', 'MASK_CHECK_OK', 'VIEW_CHECK_OK', 'MAX_LENGTH')})")
# エラーが呼び出し側で例外になること
try:
    k.run("raise ValueError('expected test error')")
    raise SystemExit("error was not propagated")
except RuntimeError as e:
    assert "expected test error" in str(e), e

# ---- 各 GPU プロファイルの依存解決（GPU 不要）: Colab と同じ Linux x86_64 / Python 3.12 / CUDA 版 torch の index で pin が解けるか
import subprocess, tempfile
for prof_name, prof in g["GPU_PROFILES"].items():
    pins = [prof["pin_overrides"].get(p.split("==")[0].split("[")[0].lower(), p) for p in g["PIP_PINNED"]] + g["KERNEL_PACKAGES"]
    req = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    req.write("\n".join(prof["torch"] + pins) + "\n"); req.close()
    idx = ["--index-url", prof["index_url"], "--extra-index-url", "https://pypi.org/simple", "--index-strategy", "unsafe-best-match"] if prof["index_url"] else []
    r = subprocess.run(g["UV"] + ["pip", "compile", "--quiet", "--python-version", "3.12", "--python-platform", "x86_64-manylinux_2_28", req.name] + idx,
                       capture_output=True, text=True, env=g["_uv_env"])
    assert r.returncode == 0, f"profile {prof_name}: pins do not resolve\n{r.stderr[-2000:]}"
    lock = {l.split("==")[0]: l.split("==")[1].split()[0] for l in r.stdout.splitlines() if "==" in l and not l.startswith("#")}
    print(f"profile {prof_name}: resolves (torch {lock.get('torch')}, sympy {lock.get('sympy')}, numpy {lock.get('numpy')}, nvidia-nccl-cu12 {lock.get('nvidia-nccl-cu12')})")

# ---- 論文の数学評価用の環境（vLLM 0.8.5）の依存解決（GPU 不要）
req = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
req.write("\n".join(g["EVAL_PIP_PINNED"]) + "\n"); req.close()
r = subprocess.run(g["UV"] + ["pip", "compile", "--quiet", "--python-version", "3.12", "--python-platform", "x86_64-manylinux_2_28", req.name],
                   capture_output=True, text=True, env=g["_uv_env"])
assert r.returncode == 0, f"eval env pins do not resolve\n{r.stderr[-2000:]}"
lock = {l.split("==")[0]: l.split("==")[1].split()[0] for l in r.stdout.splitlines() if "==" in l and not l.startswith("#")}
print(f"eval env: resolves (vllm {lock.get('vllm')}, torch {lock.get('torch')}, xformers {lock.get('xformers')}, numpy {lock.get('numpy')})")

# ---- セル 1 の再実行: 前回の公式環境カーネルと、残った学習プロセス（模擬）を終了する
fake = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", "torch.distributed.run"], start_new_session=True)
open(f"{g['WORK_DIR']}/ao-trainer.pgid", "w").write(str(fake.pid))
old_kernel = g["AO_KERNEL"]
exec(compile(open(os.path.join(SRC, "cells", "02_env_bootstrap.py")).read(), "02_env_bootstrap.py", "exec"), g)
time.sleep(2)
assert fake.poll() is not None, "leftover trainer process group was not killed"
assert not old_kernel.alive(), "previous official-env kernel was not killed"
g["AO_KERNEL"].run("print('new kernel ok')")
print("re-run of cell 1 killed the previous kernel and trainer OK")
print("ENV BOOTSTRAP TEST PASSED")
