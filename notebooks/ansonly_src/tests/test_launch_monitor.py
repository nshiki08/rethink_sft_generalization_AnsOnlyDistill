"""CPU test of launch_training / TrainingMonitor / checkpoint completeness / kill-after-step / metric parsing using fake_trainer.

Run run_cells_local.py up to cell 05 first (it creates the real AO parquet in $AO_TEST_WORK_DIR).
Usage: python notebooks/ansonly_src/tests/test_launch_monitor.py
"""
import os, sys, json, time, glob, shutil, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(SRC))
TEST_WORK = os.path.expanduser(os.environ.get("AO_TEST_WORK_DIR", "~/.cache/ao_nb_test"))
sys.path.insert(0, REPO)
g = globals()


def run_cell(name):
    src = open(os.path.join(SRC, "cells", name)).read()
    exec(compile(src, name, "exec"), g)


ENV_BOOTSTRAP = dict(in_colab=False, work_dir=os.path.join(TEST_WORK, "ao_work_launchtest"), repo_dir=REPO)
run_cell("03_config.py")
WORK_DIR = os.path.join(TEST_WORK, "ao_work_launchtest")
shutil.rmtree(WORK_DIR, ignore_errors=True)
REPO_DIR, DATA_DIR, MODELS_DIR, CKPT_DIR, LOG_DIR, RECORD_DIR = REPO, f"{WORK_DIR}/data", f"{WORK_DIR}/models", f"{WORK_DIR}/ckpt", f"{WORK_DIR}/log", f"{WORK_DIR}/records"
for d in (DATA_DIR, MODELS_DIR, CKPT_DIR, LOG_DIR, RECORD_DIR):
    os.makedirs(d, exist_ok=True)
FORK_COMMIT = "deadbeef"; N_GPUS = 1; HF_LOGGED_IN = False; HF_ACCOUNT_NAME = None; FLASH_ATTN_OK = True
GPU_PROFILE, GPU_PROFILE_DEVIATION = "official", None
FORK_REPO_URL = "https://github.com/nshiki08/rethink_sft_generalization_AnsOnlyDistill"; UPSTREAM_REFERENCE_COMMIT = "71a442ea8f0adc4a1df4529d3c43393ac6e504fd"
FSDP2_GRAD_CHECK = dict(status="ok")
OFFICIAL_CODE_UNCHANGED = True; ENV_RECORD = {}; AO_AUDIT_SUMMARY = {}; LENGTH_RECORD = {}


def sha256_of(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


import pyarrow.parquet as pq
AO_PARQUET = os.path.join(DATA_DIR, "Math-AO-20k.parquet")
pq.write_table(pq.read_table(os.path.join(TEST_WORK, "ao_work_local", "data", "Math-AO-20k.parquet")).slice(0, 512), AO_PARQUET)   # 実データの先頭 512 行
AO_SHA256 = sha256_of(AO_PARQUET)
run_cell("06_helpers_runspec.py")
get_base_model_local = lambda: os.path.join(TEST_WORK, "tok", "Qwen3-1.7B-Base")
run_cell("09_helpers_launch_monitor_hf.py")

# --- unit: parse_metric_line
m = parse_metric_line("step:12 - train/loss:0.123 - train/lr:4.5e-05 - train/grad_norm:1.5 - train/entropy/mean:0.2")
assert m == dict(step=12, **{"train/loss": 0.123, "train/lr": 4.5e-05, "train/grad_norm": 1.5, "train/entropy/mean": 0.2}), m
assert parse_metric_line("Epoch 1/3: 50%|#####  | 1/2") is None
print("parse_metric_line OK")

# --- unit: hydra_value / fmt_lr
assert hydra_value("null") is None and hydra_value("False") is False and hydra_value("20000") == 20000 and hydra_value("5e-5") == 5e-5
assert hydra_value('["console","wandb"]') == ["console", "wandb"] and hydra_value("[0.9,0.999]") == [0.9, 0.999]
assert fmt_lr(5e-5) == "5e-5" and fmt_lr(1e-4) == "1e-4" and fmt_lr(2e-5) == "2e-5" and fmt_lr(1.5e-5) == "1.5e-5", (fmt_lr(5e-5), fmt_lr(1e-4))
print("hydra_value/fmt_lr OK")

# --- spec with fake trainer module (2 steps/epoch hard-coded in fake_trainer -> use 512 rows)
spec = build_run_spec("trial", 5e-5, 3, data_path=AO_PARQUET, data_sha256=AO_SHA256, n_rows=512, max_length=1536, save_freq=3, run_id="trial-launchtest", upload_policy="all")
assert spec["upload_steps"] == [3, 6]
spec_c = build_run_spec("baseline", 5e-5, 8, data_path=AO_PARQUET, data_sha256=AO_SHA256, n_rows=20480, max_length=1536, save_freq=10, run_id="x")
assert spec_c["upload_steps"] == [10, 20, 40, 80, 160, 240, 320, 400, 480, 560, 640], spec_c["upload_steps"]
assert spec_c["upload_full_steps"] == [80, 160, 240, 320, 400, 480, 560, 640], spec_c["upload_full_steps"]   # 10/20/40 は重みのみ
assert checkpoint_content_for(spec_c, 10) == "model_only" and checkpoint_content_for(spec_c, 80) == "full"
assert checkpoint_content_for(dict(upload_steps=[10]), 10) == "full"   # 旧版の spec
# 論文の最適化条件（16 epoch・constant）: scheduler の引数、run_id、評価 step（1280 まで）
spec_k = build_run_spec("search", 1e-4, 16, data_path=AO_PARQUET, data_sha256=AO_SHA256, n_rows=20480, max_length=1536, save_freq=10, run_id=None, lr_scheduler="constant")
assert spec_k["overrides"]["optim.lr_scheduler"] == "constant" and spec_k["run_id"].endswith("_ConstLR_search"), spec_k["run_id"]
assert spec_k["paper_condition"] == "Sec. 3.4 Setting 4" and spec_k["total_steps"] == 1280
assert [x for x in spec_k["upload_steps"] if x > 640] == [720, 800, 880, 960, 1040, 1120, 1200, 1280], spec_k["upload_steps"]
assert any(c["key"] == "optim.lr_scheduler" for c in spec_k["changes_vs_official"])
assert spec_c["overrides"]["optim.lr_scheduler"] == "cosine" and spec_c["paper_condition"].startswith("default")
assert build_run_spec("search", 2e-5, 4, data_path=AO_PARQUET, data_sha256=AO_SHA256, n_rows=20480, max_length=1536, save_freq=10, run_id="y")["paper_condition"] is None
assert normalize_run((5e-5, 16)) == (5e-5, 16, "cosine") and normalize_run((1e-4, 16, "constant")) == (1e-4, 16, "constant")
assert spec["upload_full_steps"] == [3, 6]   # 試走は "all"
require_training_env(1)
FSDP2_GRAD_CHECK = dict(status="mismatch")
try:
    require_training_env(1); raise SystemExit("grad check gate did not fire")
except AssertionError:
    pass
FSDP2_GRAD_CHECK = dict(status="ok")
assert len(spec_c["expected_save_steps"]) == 64
assert spec["steps_per_epoch"] == 2 and spec["total_steps"] == 6 and spec["expected_save_steps"] == [3, 6], spec["expected_save_steps"]
assert spec["overrides"]["data.response_key"] == "teacher_answer" and spec["overrides"]["trainer.checkpoint.save_contents"] == '["model","optimizer","extra"]'
assert spec["overrides"]["trainer.resume_mode"] == "disable" and spec["overrides"]["optim.lr"] == "5e-5" and spec["overrides"]["trainer.total_epochs"] == "3"
assert "trainer.total_training_steps" not in spec["overrides"]
# --- official 8-GPU emulation on 1 GPU
v = spec["training_view"]
assert v["enabled"] and v["advantage"] == 0.125 and spec["logged_loss_scale"] == 0.125
assert spec["overrides"]["trainer.importance_sampling_mode"] == "adv-only"
assert spec["overrides"]["data.train_files"] == spec["train_file"] != AO_PARQUET and spec["data_path"] == AO_PARQUET
_rows = pq.read_table(spec["train_file"], columns=["ao_source_row"]).column("ao_source_row").to_pylist()
assert official_micro_batches(_rows, 1, 256, 4) == official_micro_batches(list(range(512)), 8, 256, 4)
assert official_micro_batches(list(range(512)), 1, 256, 4) != official_micro_batches(list(range(512)), 8, 256, 4)
for n2 in (2, 4):
    p2, s2, v2 = prepare_train_file(AO_PARQUET, AO_SHA256, n2)
    r2 = pq.read_table(p2, columns=["ao_source_row"]).column("ao_source_row").to_pylist()
    assert v2["advantage"] == n2 / 8 and official_micro_batches(r2, n2, 256, 4) == official_micro_batches(list(range(512)), 8, 256, 4)
p8, s8, v8 = prepare_train_file(AO_PARQUET, AO_SHA256, 8)
assert not v8["enabled"] and p8 == AO_PARQUET
print("emulation checks OK; residual diffs:", len(spec["residual_differences"]))
print("changes_vs_official:", [(c["key"], c["new"]) for c in spec["changes_vs_official"]])
spec["module"] = "fake_trainer"
os.environ["PYTHONPATH"] = os.path.join(HERE, "fake_trainer")

# --- unit: nan metrics keep the step
m = parse_metric_line("step:5 - train/loss:nan - train/lr:1e-05 - train/grad_norm:inf")
assert m["step"] == 5 and m["train/lr"] == 1e-05 and m["train/loss"] != m["train/loss"], m

# --- run A: kill after step 3 (upload disabled -> pause + kill when local checkpoint complete)
resA = launch_training(spec, kill_after_step=3, upload=False, poll_sec=1, stable_scans=False)
assert resA["killed_by_monitor"], resA
stepsA = [m["step"] for m in resA["metrics"]]
assert stepsA[0] == 1 and 3 in resA["monitor"]["complete_local"], resA["monitor"]
assert max(stepsA) < spec["total_steps"], f"run A must be interrupted before the end: {stepsA}"
assert 6 not in resA["monitor"]["complete_local"]
ok, sizes = checkpoint_is_complete(f"{CKPT_DIR}/trial-launchtest/global_step_3", 1, require_stable=False)
assert ok, sizes
man = build_manifest(spec, 3, f"{CKPT_DIR}/trial-launchtest/global_step_3")
assert all("sha256" in f for f in man["files"]) and man["run_config"]["run_id"] == "trial-launchtest"
tA = summarize_timing(resA)
print("timing A:", {k: v for k, v in tA.items() if k != "step_sec_all"})

# --- resume B in a new process from the local step-3 checkpoint (no HF): build_resume_spec path
man["folder_commit"] = "nocommit"
json.dump(dict(run_id="trial-launchtest", step=3, folder_commit="nocommit"), open(f"{CKPT_DIR}/trial-launchtest/global_step_3/ao_upload_verified.json", "w"))
MAX_LENGTH = 1536
probs, warns = check_resume_compatibility(man, 1)
assert not probs, probs
probs2, _ = check_resume_compatibility(man, 2)
assert any("world_size" in p for p in probs2), probs2
_saved_flag = EMULATE_OFFICIAL_WORLD_SIZE
EMULATE_OFFICIAL_WORLD_SIZE = False
probs3, _ = check_resume_compatibility(man, 1)
assert any("公式 8 GPU 再現" in p for p in probs3), probs3
EMULATE_OFFICIAL_WORLD_SIZE = _saved_flag
specB = build_resume_spec(man, f"{CKPT_DIR}/trial-launchtest/global_step_3")
assert specB["overrides"]["trainer.resume_mode"] == "resume_path" and specB["overrides"]["trainer.total_epochs"] == "3"
assert specB["overrides"]["data.train_files"] == spec["train_file"] and specB["overrides"]["trainer.importance_sampling_mode"] == "adv-only"
assert specB["data_path"] == AO_PARQUET and specB["training_view"]["order_sha256"] == spec["training_view"]["order_sha256"]
assert abs(resA["metrics"][0]["train/loss_official_scale"] - resA["metrics"][0]["train/loss"] / 0.125) < 1e-12
specB["module"] = "fake_trainer"
resB = launch_training(specB, upload=False, poll_sec=1)
stepsB = [m["step"] for m in resB["metrics"]]
assert resB["exit_code"] == 0 and stepsB == [4, 5, 6], stepsB
logB = open(resB["log_path"]).read()
assert "Loaded optimizer from" in logB and "Successfully loaded dataloader state" in logB
assert 6 in resB["monitor"]["complete_local"]
assert resB["monitor"]["uploaded"].get(3, {}).get("previously_verified"), resB["monitor"]["uploaded"]
assert not resB["monitor"]["failed"]
assert os.path.isdir(f"{CKPT_DIR}/trial-launchtest/global_step_3"), "resume source must not be deleted"
print("resume B OK; lr trace:", [(m["step"], m["train/lr"]) for m in resB["metrics"]])

# --- incomplete checkpoint must not be considered complete
d = f"{CKPT_DIR}/trial-launchtest/global_step_9"
os.makedirs(d + "/huggingface", exist_ok=True)
open(d + "/model_world_size_1_rank_0.pt", "wb").write(b"PK\x03\x04partial")  # truncated zip
ok, _ = checkpoint_is_complete(d, 1, require_stable=False)
assert not ok
print("incomplete checkpoint rejected OK")

# --- model_only (重みのみ) checkpoint: optimizer 状態を除いて manifest を作り、完全性は content 別に判定する
d_mo = f"{CKPT_DIR}/trial-launchtest/global_step_3_model_only_copy/global_step_3"
shutil.copytree(f"{CKPT_DIR}/trial-launchtest/global_step_3", d_mo)
open(os.path.join(os.path.dirname(d_mo), "latest_checkpointed_iteration.txt"), "w").write("3")
man_mo = build_manifest(spec, 3, d_mo, content="model_only")
assert man_mo["content"] == "model_only" and not any(f["path"].startswith("optim_") for f in man_mo["files"])
for f in glob.glob(os.path.join(d_mo, "optim_world_size_*")):
    os.remove(f)
assert checkpoint_is_complete(d_mo, 1, require_stable=False, content="model_only")[0]
assert not checkpoint_is_complete(d_mo, 1, require_stable=False, content="full")[0]
pr_mo, _ = check_resume_compatibility(man_mo, 1)
assert any("重みのみ" in p for p in pr_mo), pr_mo
man_gp = json.loads(json.dumps(man)); man_gp["run_config"]["gpu_profile"] = "blackwell"
pr_gp, _ = check_resume_compatibility(man_gp, 1)
assert any("GPU プロファイル" in p for p in pr_gp), pr_gp
print("model_only checkpoint / GPU profile checks OK")

# --- upload policy: non-selected steps are deleted locally and never uploaded
spec_p = build_run_spec("trial", 5e-5, 3, data_path=AO_PARQUET, data_sha256=AO_SHA256, n_rows=512, max_length=1536, save_freq=3, run_id="trial-policy")
assert spec_p["upload_steps"] == [6], spec_p["upload_steps"]
spec_p["module"] = "fake_trainer"
resP = launch_training(spec_p, upload=False, poll_sec=1, quiet=True)
assert resP["exit_code"] == 0 and resP["monitor"]["skipped"] == [3], resP["monitor"]
assert not os.path.isdir(f"{CKPT_DIR}/trial-policy/global_step_3") and os.path.isdir(f"{CKPT_DIR}/trial-policy/global_step_6")
print("upload policy OK")

# --- legacy checkpoint (no training_view): blocked with emulation on, allowed with emulation off
man_old = json.loads(json.dumps(man)); rc = man_old["run_config"]
for k in ("training_view", "train_file", "train_file_sha256", "logged_loss_scale", "upload_steps", "upload_policy", "residual_differences", "official_world_size"):
    rc.pop(k, None)
rc["overrides"]["trainer.importance_sampling_mode"] = "vanilla"; rc["overrides"]["data.train_files"] = AO_PARQUET
pr_old, w_old = check_resume_compatibility(man_old, 1)
assert any("公式 8 GPU 再現" in p for p in pr_old) and any("training_view" in w for w in w_old), (pr_old, w_old)
EMULATE_OFFICIAL_WORLD_SIZE = False
pr_old2, _ = check_resume_compatibility(man_old, 1)
assert not pr_old2, pr_old2
sp_old = build_resume_spec(man_old, f"{CKPT_DIR}/trial-launchtest/global_step_3")
assert sp_old["overrides"]["data.train_files"] == AO_PARQUET and sp_old["logged_loss_scale"] == 1.0 and sp_old["upload_policy"] == "all"
EMULATE_OFFICIAL_WORLD_SIZE = True
print("legacy resume OK")

rec = write_experiment_record(spec, [resA, resB], upload=False)
assert rec["runs"][1]["first_step"] == 4
print("ALL LAUNCH/MONITOR TESTS PASSED")
