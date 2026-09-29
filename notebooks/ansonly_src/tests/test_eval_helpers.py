"""CPU test of the evaluation notebook helpers (paper-protocol evaluation). vLLM generation itself needs a GPU and is not run here.

Checks: the symlink to the official hard-coded data path, that the official script's dataset loaders/keys work on it,
parsing of the official result file, paper reference lookup.
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

# 3. 論文の値
assert g["paper_reference"]("ao", 640, "MATH500", "default（Sec. 2.1, Tab. 3）") == 56.2 and g["paper_reference"]("ao", 640, "AIME24", "default（Sec. 2.1, Tab. 3）") == 5.0
assert g["paper_reference"]("ao", 640, "MATH500", "Sec. 3.4 Setting 4") is None and g["paper_reference"]("ao", 640, "MATH500", None) is None   # 既定条件以外には並べない
assert g["paper_reference"]("cot", 640, "MATH500") == 56.2
assert g["paper_reference"]("base", 0, "MATH500") == 58.9 and g["paper_reference"]("ao", 30, "MATH500") is None
_mk = g["MODEL_KEY"]
g["MODEL_KEY"] = "Qwen3-8B"
assert g["paper_reference"]("ao", 640, "AIME24", "default（Sec. 2.1, Tab. 3）") == 47.7 and g["paper_reference"]("base", 0, "MATH500") == 76.8
g["MODEL_KEY"] = "Qwen3-14B"
assert g["paper_reference"]("ao", 640, "MATH500", "default（Sec. 2.1, Tab. 3）") == 95.1 and g["paper_reference"]("base", 0, "AIME24") == 14.7
g["MODEL_KEY"] = _mk
# 4. 評価ノートブック: run_id 未指定なら HF にある MODEL_KEY の本学習 run を評価対象にする（試走と別モデルは除く）
class _FakeApi:
    def list_repo_files(self, repo_id, repo_type="model"):
        return ["README.md", "runs/Qwen3-1.7B_Math-AO-20k_lr5e-5_ep8_bs256_baseline/global_step_10/ao_ckpt_manifest.json",
                "runs/Qwen3-1.7B_Math-AO-20k_lr5e-5_ep8_bs256_baseline/paper_eval/x.json",
                "runs/trial-Qwen3-1.7B_x/global_step_3/ao_ckpt_manifest.json", "runs/Qwen3-4B_Math-AO-20k_lr5e-5_ep8_bs256_baseline/global_step_10/a.json"]
g.update(HF_API=_FakeApi(), HF_CKPT_REPO_ID="me/rethink-sft-ao-checkpoints", hf_repo_exists=lambda r, repo_type="model": True, MODEL_KEY="Qwen3-1.7B",
         PAPER_EVAL_AO_RUNS=None, TRAIN_RESULTS={},
         list_hf_checkpoints=lambda rid: [dict(step=s, complete=True) for s in (10, 20, 30, 40, 640)],
         get_manifest=lambda rid, step: {"run_config": {"paper_condition": "default（Sec. 2.1, Tab. 3）"}})
assert g["hf_model_runs"]() == ["Qwen3-1.7B_Math-AO-20k_lr5e-5_ep8_bs256_baseline"]
_t = g["paper_eval_targets"]()
assert [(t["kind"], t["step"]) for t in _t] == [("ao", 10), ("ao", 20), ("ao", 40), ("ao", 640)], _t   # 論文の評価 step のうち HF にあるもの
assert g["PAPER_EVAL_K"] == {"MATH500": 3, "AIME24": 10}
print("eval-notebook run selection OK")
print("ALL EVAL HELPER TESTS PASSED")
