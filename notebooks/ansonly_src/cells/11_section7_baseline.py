# @title 7. baseline（公式 CoT 設定 lr 5e-5 / 8 epoch の AO 学習）の確認または実行
TRAIN_RESULTS = globals().get("TRAIN_RESULTS", {})


def preflight_train(spec):
    """本学習の直前確認: データ条件・ゲート・試走設定の混入・衝突"""
    assert RUN_MODE == "train", "RUN_MODE='train' でのみ本学習を開始する"
    require_training_env(spec["n_gpus"])
    assert AO_DATA_READY, "AO データに未解決の抽出/照合問題がある（セクション 3）"
    assert MASK_CHECK_OK, "loss mask の確認に失敗している（セクション 4）"
    assert spec["data_path"] == AO_PARQUET and spec["n_rows"] == DATASET_EXPECTED_ROWS and spec["data_sha256"] == AO_SHA256, "本学習は全 20,480 行の AO データで行う"
    assert not spec["run_id"].startswith("trial-") and spec["total_training_steps_override"] is None, "試走設定が本学習に混入している"
    assert "trainer.total_training_steps" not in spec["overrides"], "step 上限は本学習で使わない"
    assert spec["overrides"]["trainer.resume_mode"] == "disable", "新規 run は同じ Base から独立に開始する（再開は『再開』セル）"
    if spec["n_gpus"] != OFFICIAL_WORLD_SIZE:
        if spec["training_view"]["enabled"]:
            assert VIEW_CHECK_OK and TRAIN_VIEW_CHECK.get("actual_world") == spec["n_gpus"], \
                "公式 8 GPU の再現確認（セクション 4-b の 4.4）が未実施または失敗。同じ GPU 台数で 4-b を実行し直す"
            assert spec["training_view"]["order_sha256"] == TRAIN_VIEW_CHECK["order_sha256"]
        else:
            print(f"*** 注意: EMULATE_OFFICIAL_WORLD_SIZE=False。GPU {spec['n_gpus']} 台では micro batch 構成と勾配の大きさが公式 CoT 学習と異なる ***")
    if HF_UPLOAD_CHECKPOINTS:
        assert HF_API is not None, "HF 未ログイン。学習中の HF 保存が要件なので開始しない"
        remote = hf_run_paths(HF_CKPT_REPO_ID, spec["run_id"])
        assert not remote, f"HF に同名 run {spec['run_id']} が存在する（steps {sorted(remote)}）。上書きしない。再開は『再開』セル、別 run は RUN_ID_SUFFIX"
    assert not os.path.exists(spec["overrides"]["trainer.default_local_dir"]), f"ローカルに同名 run のディレクトリがある: {spec['overrides']['trainer.default_local_dir']}"
    _disk = shutil.disk_usage(WORK_DIR)
    print(f"preflight OK: disk free {_disk.free / 1e9:.0f} GB (ckpt ≈ {CKPT_SIZE_EST_GB:.1f} GB x local keep {LOCAL_KEEP_LAST_N_VERIFIED_CKPTS})")


if RUN_MODE != "train" or RUN_KIND != "baseline":
    print("RUN_MODE='train' かつ RUN_KIND='baseline' のときだけ実行する（現在: %s / %s）" % (RUN_MODE, RUN_KIND))
    if HF_API is not None and HF_CKPT_REPO_ID:
        _bid = make_run_id("baseline", OFFICIAL_BASELINE["lr"], OFFICIAL_BASELINE["epochs"], suffix=RUN_ID_SUFFIX)
        _remote = hf_run_paths(HF_CKPT_REPO_ID, _bid)
        print(f"HF 上の baseline run {_bid}: {'steps ' + str(sorted(_remote)) if _remote else 'なし'}")
        print("  注意: 既存成果物があっても、その run_config（manifest）で AO target と設定を確認するまで再利用可能とは判断しない")
else:
    BASELINE_SPEC = build_run_spec("baseline", OFFICIAL_BASELINE["lr"], OFFICIAL_BASELINE["epochs"], data_path=AO_PARQUET, data_sha256=AO_SHA256,
                                   n_rows=DATASET_EXPECTED_ROWS, max_length=MAX_LENGTH, save_freq=SAVE_FREQ,
                                   extra_note="baseline: 公開 CoT 設定 (lr 5e-5, 8 epoch) を AO データで実行")
    print_run_spec(BASELINE_SPEC)
    preflight_train(BASELINE_SPEC)
    TRAIN_RESULTS[BASELINE_SPEC["run_id"]] = launch_training(BASELINE_SPEC)
    _r = TRAIN_RESULTS[BASELINE_SPEC["run_id"]]
    print("timing:", json.dumps({k: v for k, v in summarize_timing(_r).items() if k != "step_sec_all"}, default=str))
    write_experiment_record(BASELINE_SPEC, [_r], unverified=["論文と同じ評価（評価ノートブック）未実施", "最終モデル未変換"])
