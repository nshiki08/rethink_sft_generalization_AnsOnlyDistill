# @title 再開: Colab 切断後に HF の完了済み checkpoint から再開する（RESUME_RUN_ID を設定。セル 0〜4 と 6-a を先に実行）
# 手順: 1) 環境と Fork の commit を復元（セル 2） 2) HF の checkpoint 一覧 3) 最新の完了済み or 指定 step を選択
#       4) HF revision を記録して取得・manifest で検証 5) trainer.resume_mode=resume_path / resume_from_path で再開
TRAIN_RESULTS = globals().get("TRAIN_RESULTS", {})
if not RESUME_RUN_ID:
    print("RESUME_RUN_ID が未設定。再開しない。HF 上の run 一覧:")
    if HF_API is not None and HF_CKPT_REPO_ID and hf_repo_exists(HF_CKPT_REPO_ID):
        _runs = sorted({f.split("/")[1] for f in HF_API.list_repo_files(HF_CKPT_REPO_ID, repo_type="model") if f.startswith("runs/")})
        for _r in _runs:
            print("  ", _r)
else:
    assert RUN_MODE in ("train", "trial"), "再開は RUN_MODE='train'（本学習）または 'trial'（試走 run）で行う"
    assert AO_DATA_READY and MASK_CHECK_OK
    TRAIN_RESULTS[RESUME_RUN_ID] = resume_run_from_hf(RESUME_RUN_ID, step=RESUME_STEP)
    _r = TRAIN_RESULTS[RESUME_RUN_ID]
    print("timing:", json.dumps({k: v for k, v in summarize_timing(_r).items() if k != "step_sec_all"}, default=str))
    write_experiment_record(_r["spec"], [_r], unverified=["再開 run。それ以前の run 結果は HF の experiment_record を参照"])
