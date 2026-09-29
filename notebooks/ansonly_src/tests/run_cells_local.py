"""Execute notebook cells locally (CPU, no HF token) in order with minimal stubs, to validate the non-GPU path.

Usage: python notebooks/ansonly_src/tests/run_cells_local.py [up_to_cell_prefix]
Work dir: $AO_TEST_WORK_DIR (default ~/.cache/ao_nb_test). Data and the tokenizer are downloaded there; nothing is written to the repo.
Run it with a Python that has the official pins installed (CPU torch is fine); that interpreter plays the "official env kernel".
Cells executed: 01_auth, (02 env bootstrap replaced by a stub ENV_BOOTSTRAP for this interpreter), 03_config, 04 .. 15 (skip branches).
The bootstrap cell itself is tested by test_env_bootstrap.py.
"""
import os, sys, glob, json, time, platform, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)                                   # notebooks/ansonly_src
REPO = os.path.dirname(os.path.dirname(SRC))                   # repo root
TEST_WORK = os.path.expanduser(os.environ.get("AO_TEST_WORK_DIR", "~/.cache/ao_nb_test"))
os.makedirs(TEST_WORK, exist_ok=True)
os.chdir(TEST_WORK)
sys.path.insert(0, REPO)
up_to = sys.argv[1] if len(sys.argv) > 1 else "99"

g = globals()   # cells run in the real __main__ so multiprocessing (fork) can resolve cell-defined functions
cells = sorted(glob.glob(os.path.join(SRC, "cells", "*.py")))


def stub_model_dir():
    """tokenizer と config だけを取得する（重みは取得しない）"""
    d = os.path.join(TEST_WORK, "tok", "Qwen3-1.7B-Base")
    if not os.path.isfile(os.path.join(d, "tokenizer.json")):
        from huggingface_hub import snapshot_download
        snapshot_download("Qwen/Qwen3-1.7B-Base", revision=g["MODEL_INFO"]["base_revision"], local_dir=d,
                          allow_patterns=["*.json", "*.txt", "merges.txt"])
    return d


def stub_env_bootstrap():
    """セル 1 の代わり: この Python を公式環境カーネルとみなした ENV_BOOTSTRAP"""
    import importlib
    vers = {}
    for m in ("torch", "transformers", "numpy", "ray", "math_verify", "huggingface_hub", "pyarrow"):
        try:
            vers[m] = getattr(importlib.import_module(m), "__version__", "n/a")
        except ImportError:
            vers[m] = "not installed"
    import torch
    return dict(
        kernel_python=platform.python_version(), train_python=platform.python_version(), train_env_dir=sys.prefix, train_env_hash="local",
        gpu_profile="official", gpu_profile_deviation=None, torch_pins=["torch==2.6.0"], torch_index_url=None,
        flash_attn_version="2.7.4.post1", flash_attn_wheel=None, pip_pinned=[], gpus=[],
        torch_probe=dict(python=platform.python_version(), torch=torch.__version__, cuda=None, cuda_available=False, arch_list=[], cap=None, cudnn=None, nccl=None),
        packages=vers, in_colab=False, work_dir=os.path.join(TEST_WORK, "ao_work_local"), repo_dir=REPO,
        fork_repo="https://github.com/nshiki08/rethink_sft_generalization_AnsOnlyDistill", fork_ref="local",
        fork_commit=subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        upstream_repo="https://github.com/Nebularaid2000/rethink_sft_generalization",
        upstream_reference_commit="71a442ea8f0adc4a1df4529d3c43393ac6e504fd", upstream_fetched=True,
        official_code_unchanged=True, official_diff_stat="",
        uv_cmd=["uv"], uv_cache_dir="", uv_python_install_dir="", eval_env_dir=os.path.join(TEST_WORK, "ao_work_local", "eval_env"), eval_pip_pinned=[],
    )


def run(path):
    name = os.path.basename(path)
    src = open(path).read()
    t0 = time.time()
    print(f"\n{'#' * 30} RUN {name} {'#' * 30}", flush=True)
    exec(compile(src, name, "exec"), g)
    print(f"{'#' * 30} DONE {name} in {time.time() - t0:.1f}s", flush=True)


for path in cells:
    name = os.path.basename(path)
    if name[:2] > up_to:
        break
    if name.startswith("02_"):
        g["ENV_BOOTSTRAP"] = stub_env_bootstrap()
        print("\n[stub] 02 env bootstrap replaced: this interpreter =", sys.executable)
        continue
    run(path)
    if name.startswith("06_"):
        g["get_base_model_local"] = stub_model_dir

print("\nLOCAL RUN FINISHED.", {k: g.get(k) for k in ("AO_DATA_READY", "MASK_CHECK_OK", "MAX_LENGTH", "AUTO_FIT_MAX_LENGTH", "VIEW_CHECK_OK")})
