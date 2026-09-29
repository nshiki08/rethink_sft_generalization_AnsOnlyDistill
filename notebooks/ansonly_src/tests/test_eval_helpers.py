"""CPU test of section 9 helpers (paper-protocol evaluation). vLLM generation itself needs a GPU and is not run here.

Checks: the symlink to the official hard-coded data path, that the official script's dataset loaders/keys work on it,
parsing of the official result file, the dev scorer (same as process_single_result_no_truncation), paper reference lookup.
Usage: python notebooks/ansonly_src/tests/test_eval_helpers.py   (Python with the official pins; creates /mnt/shared-storage-user/... link)
"""
import os, sys, json, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(SRC))
TEST_WORK = os.path.expanduser(os.environ.get("AO_TEST_WORK_DIR", "~/.cache/ao_nb_test"))
sys.path.insert(0, REPO)
g = {"__name__": "__main__"}
g["ENV_BOOTSTRAP"] = dict(in_colab=False, work_dir=os.path.join(TEST_WORK, "ao_work_evaltest"), repo_dir=REPO,
                          eval_env_dir=os.path.join(TEST_WORK, "ao_work_evaltest", "eval_env"), eval_pip_pinned=[], uv_cmd=["uv"],
                          uv_cache_dir="", uv_python_install_dir="")
exec(compile(open(os.path.join(SRC, "cells", "03_config.py")).read(), "03_config.py", "exec"), g)
g.update(GPU_PROFILE="official", N_GPUS=0, HF_API=None, HF_CKPT_REPO_ID=None, TRAIN_RESULTS={}, RECORD_DIR=os.path.join(TEST_WORK, "ao_work_evaltest", "records"))
exec(compile(open(os.path.join(SRC, "cells", "14_section9_eval.py")).read(), "14_section9_eval.py", "exec"), g)

# 1. 公式スクリプトの絶対パスへのリンクと、公式のローダー・キーでデータが読めること
g["ensure_official_eval_paths"]()
assert os.path.realpath(g["OFFICIAL_EVAL_ROOT"]) == os.path.realpath(REPO)
sys.path.insert(0, os.path.join(REPO, "evaluation", "math_eval"))
from datasets import load_dataset, load_from_disk
os.chdir(os.path.join(REPO, "evaluation", "math_eval"))
from utils import DATASET_KEYS   # evaluation/math_eval/utils/utils.py（公式）
paths = g["PAPER_EVAL_DATASET_PATHS"]
assert set(paths.values()) <= set(DATASET_KEYS), "公式の DATASET_KEYS に無いパスを渡している"
m = load_dataset(paths["MATH500"])["test"]
a = load_from_disk(paths["AIME24"])["test"]
assert len(m) == 500 and len(a) == 30, (len(m), len(a))
assert DATASET_KEYS[paths["MATH500"]] == {"question": "problem", "answer": "solution"}
print("official data paths OK: MATH500", len(m), "AIME24", len(a))

# 2. 公式の結果ファイル（results/<name>/<save>_budget<B>/no_instruct_512.json）の読み取り
with tempfile.TemporaryDirectory() as d:
    p = f"{d}/results/mergedmerged_step10/math500_budget2000"
    os.makedirs(p)
    json.dump({"native": {"pass@1": 0.5, "pass@k(majority)": 0.6, "average_pass_rate": 0.55, "average_length": 12.0}, "200": {}}, open(f"{p}/no_instruct_512.json", "w"))
    r = g["read_official_math_result"](d, "MATH500")
    assert abs(r["avg@3"] - 55.0) < 1e-9 and r["avg_length_tokens"] == 12.0, r
    assert g["read_official_math_result"](d, "AIME24") is None
print("result parsing OK")

# 3. dev の採点は公式と同じ（gold を \boxed{} で包み、応答全体を parse して verify）
ns = {}
exec(compile(g["DEV_RUNNER_SRC"], "dev_runner", "exec"), ns)
assert ns["score"]("4", ["so the answer is \\boxed{4}", "\\boxed{5}", "no box"]) == [True, False, False]
assert ns["score"]("\\frac{1}{2}", ["\\boxed{0.5}"]) == [True]
print("dev scorer OK")

# 4. 論文の値
assert g["paper_reference"]("ao", 640, "MATH500", "default（Sec. 2.1, Tab. 3）") == 56.2 and g["paper_reference"]("ao", 640, "AIME24", "default（Sec. 2.1, Tab. 3）") == 5.0
assert g["paper_reference"]("ao", 640, "MATH500", "Sec. 3.4 Setting 4") is None and g["paper_reference"]("ao", 640, "MATH500", None) is None   # 既定条件以外には並べない
assert g["paper_reference"]("cot", 640, "MATH500") == 56.2
assert g["paper_reference"]("base", 0, "MATH500") == 58.9 and g["paper_reference"]("ao", 30, "MATH500") is None
print("ALL EVAL HELPER TESTS PASSED")
