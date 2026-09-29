"""Execute notebook cells locally (CPU, no HF token) in order with minimal stubs, to validate the non-GPU path.

Usage: python notebooks/ansonly_src/tests/run_cells_local.py [up_to_cell_prefix]
Work dir: $AO_TEST_WORK_DIR (default ~/.cache/ao_nb_test). Data and the tokenizer are downloaded there; nothing is written to the repo.
Cells executed: 01_auth, 02_config, 03_section1, (04 skipped: uses real repo dir instead), 05_section3_data,
06_helpers_runspec, 07_section4_length_mask, 08_section5_dryrun, 09_helpers (definitions only), 10..15 (skip branches).
"""
import os, sys, glob, json, time, types, traceback

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

# stub: make get_base_model_local() return the local tokenizer/config dir (no 3.4GB download)
def stub_model_dir():
    """tokenizer と config だけを取得する（重みは取得しない）"""
    d = os.path.join(TEST_WORK, "tok", "Qwen3-1.7B-Base")
    if not os.path.isfile(os.path.join(d, "tokenizer.json")):
        from huggingface_hub import snapshot_download
        snapshot_download("Qwen/Qwen3-1.7B-Base", revision=g["MODEL_INFO"]["base_revision"], local_dir=d,
                          allow_patterns=["*.json", "*.txt", "merges.txt"])
    return d


def run(path):
    name = os.path.basename(path)
    src = open(path).read()
    t0 = time.time()
    print(f"\n{'#' * 30} RUN {name} {'#' * 30}", flush=True)
    exec(compile(src, name, "exec"), g)
    print(f"{'#' * 30} DONE {name} in {time.time() - t0:.1f}s", flush=True)


for H_path in list(cells):
    path, name = H_path, os.path.basename(H_path)
    if name[:2] > up_to:
        break
    if name.startswith("04_"):
        # replace clone/env cell with local equivalents
        g.update(dict(REPO_DIR=REPO, FORK_COMMIT=os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(), UPSTREAM_FETCHED=True,
                      OFFICIAL_DIFF_STAT="", OFFICIAL_CODE_UNCHANGED=True, GPUS=[], N_GPUS=0, TORCH_PROBE={}, FLASH_ATTN_OK=False,
                      PKG_VERSIONS={}, ENV_RECORD={"local_stub": True}))
        import shutil, subprocess
        g["sh"] = lambda cmd, cwd=None, check=True, capture=True, env=None: subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=True, text=True).stdout.strip()
        print("\n[stub] 04 skipped: REPO_DIR =", REPO)
        continue
    if name.startswith("02_"):
        run(path)
        g["WORK_DIR"] = os.path.join(TEST_WORK, "ao_work_local")
        for k in ("REPO_DIR", "DATA_DIR", "MODELS_DIR", "CKPT_DIR", "LOG_DIR", "RECORD_DIR"):
            g[k] = os.path.join(g["WORK_DIR"], {"REPO_DIR": "repo", "DATA_DIR": "data", "MODELS_DIR": "models", "CKPT_DIR": "ckpt", "LOG_DIR": "log", "RECORD_DIR": "records"}[k])
            os.makedirs(g[k], exist_ok=True)
        g["REPO_DIR"] = REPO
        continue
    if name.startswith("06_"):
        run(path)
        g["get_base_model_local"] = stub_model_dir
        continue
    run(path)

print("\nLOCAL RUN FINISHED. keys:", [k for k in ("AO_DATA_READY", "MASK_CHECK_OK", "MAX_LENGTH", "AUTO_FIT_MAX_LENGTH") if k in g], {k: g.get(k) for k in ("AO_DATA_READY", "MASK_CHECK_OK", "MAX_LENGTH", "AUTO_FIT_MAX_LENGTH")})
