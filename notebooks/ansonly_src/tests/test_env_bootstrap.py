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
print("ENV BOOTSTRAP TEST PASSED")
