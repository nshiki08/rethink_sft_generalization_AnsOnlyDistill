# @title 8. LR / epoch / scheduler 探索（提案は論文の最適化条件。各候補を同じ Base から独立に学習。8 epoch run の途中 checkpoint を短い epoch 設定の代用にしない）
TRAIN_RESULTS = globals().get("TRAIN_RESULTS", {})
SEARCH_SPECS = {}
if RUN_MODE != "train" or RUN_KIND != "search":
    print("RUN_MODE='train' かつ RUN_KIND='search' のときだけ実行する（現在: %s / %s）" % (RUN_MODE, RUN_KIND))
    print("提案グリッド:", SEARCH_GRID_PROPOSED, "| 確定グリッド:", SEARCH_GRID_CONFIRMED, "| 実行リスト:", SEARCH_RUN_LIST)
elif not SEARCH_RUN_LIST:
    print("SEARCH_RUN_LIST が空。実行する (lr, epochs, scheduler) を設定セルで指定する")
else:
    _confirmed = {normalize_run(c) for c in (SEARCH_GRID_CONFIRMED or [])}
    for _lr, _ep, _sched in (normalize_run(r) for r in SEARCH_RUN_LIST):
        _tag = "confirmed" if (_lr, _ep, _sched) in _confirmed else "proposed"
        _spec = build_run_spec("search", _lr, _ep, data_path=AO_PARQUET, data_sha256=AO_SHA256, n_rows=DATASET_EXPECTED_ROWS, max_length=MAX_LENGTH,
                               save_freq=SAVE_FREQ, lr_scheduler=_sched,
                               extra_note=f"探索候補 ({_tag})。総 epoch {_ep} に応じて scheduler 期間 = {steps_per_epoch(DATASET_EXPECTED_ROWS, OFFICIAL_TBS, N_GPUS) * _ep} step")
        SEARCH_SPECS[_spec["run_id"]] = _spec
    for _rid, _spec in SEARCH_SPECS.items():
        print("=" * 100)
        print_run_spec(_spec)
        _remote = hf_run_paths(HF_CKPT_REPO_ID, _rid) if HF_API else {}
        if _remote:
            print(f"HF に {_rid} が既にある (steps {sorted(_remote)})。完了済みなら再実行しない。途中なら『再開』セルで RESUME_RUN_ID を指定する")
            continue
        preflight_train(_spec)
        TRAIN_RESULTS[_rid] = launch_training(_spec)
        write_experiment_record(_spec, [TRAIN_RESULTS[_rid]], unverified=["dev 評価未実施", "最終モデル未変換"])
    print("探索 run の完了状況:", {k: ("OK" if v["exit_code"] == 0 else f"rc={v['exit_code']}") for k, v in TRAIN_RESULTS.items()})
