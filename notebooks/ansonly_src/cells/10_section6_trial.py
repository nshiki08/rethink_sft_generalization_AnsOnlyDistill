# @title 6-b. 短い試走: 学習 → HF 保存 → プロセス終了 → HF から取得 → 新プロセスで再開（RUN_MODE="trial" のときだけ実行）
import pyarrow.parquet as pq, json, os, math, numpy as np

TRIAL_RESULTS = {}
if RUN_MODE != "trial":
    print("RUN_MODE != 'trial' なのでスキップ")
else:
    require_training_env()
    assert AO_DATA_READY, "AO データが未確定（セクション 3 のゲート）。試走でも同じデータ条件を使う"
    # 試走データ: AO parquet の先頭 TRIAL_NUM_ROWS 行（行順維持）。本学習と混ざらないよう別ファイル・別 run_id
    _trial_rows = (TRIAL_NUM_ROWS // OFFICIAL_TBS) * OFFICIAL_TBS
    assert _trial_rows >= OFFICIAL_TBS, "TRIAL_NUM_ROWS は global batch 256 以上"
    TRIAL_PARQUET = f"{DATA_DIR}/trial/Math-AO-trial{_trial_rows}.parquet"
    os.makedirs(os.path.dirname(TRIAL_PARQUET), exist_ok=True)
    pq.write_table(AO_TABLE.slice(0, _trial_rows), TRIAL_PARQUET)
    TRIAL_SHA256 = sha256_of(TRIAL_PARQUET)
    _ts = time.strftime("%Y%m%d-%H%M%S")
    _trial_id = f"trial-{MODEL_KEY}_Math-AO-{_trial_rows}_lr{fmt_lr(OFFICIAL_BASELINE['lr'])}_ep{TRIAL_EPOCHS}_bs{OFFICIAL_TBS}_{_ts}"
    _spe = steps_per_epoch(_trial_rows, OFFICIAL_TBS, N_GPUS)
    _total = _spe * TRIAL_EPOCHS
    assert TRIAL_KILL_AFTER_STEP in range(TRIAL_SAVE_FREQ, _total, TRIAL_SAVE_FREQ), f"TRIAL_KILL_AFTER_STEP は保存 step（{TRIAL_SAVE_FREQ} の倍数）かつ最終 step {_total} 未満"
    TRIAL_SPEC = build_run_spec("trial", OFFICIAL_BASELINE["lr"], TRIAL_EPOCHS, data_path=TRIAL_PARQUET, data_sha256=TRIAL_SHA256, n_rows=_trial_rows,
                                max_length=MAX_LENGTH, save_freq=TRIAL_SAVE_FREQ, upload_policy="all", run_id=_trial_id,
                                extra_note=f"試走。データ {_trial_rows} 行 / {TRIAL_EPOCHS} epoch / step 上限なし（scheduler 期間 = {_total} step）。本学習の設定ではない")
    print_run_spec(TRIAL_SPEC)
    if TRIAL_SPEC["training_view"]["enabled"]:
        # 試走 subset でも、各 step の micro batch が公式 8 GPU と同じ 4 行の組になることを確認（並べ替えは 256 行 block 内なので本学習の先頭 step とも同じ）
        _tv_rows = pq.read_table(TRIAL_SPEC["train_file"], columns=["ao_source_row"]).column("ao_source_row").to_pylist()
        assert official_micro_batches(_tv_rows, N_GPUS, OFFICIAL_TBS, OFFICIAL_MICRO_BSZ) == \
            official_micro_batches(list(range(_trial_rows)), OFFICIAL_WORLD_SIZE, OFFICIAL_TBS, OFFICIAL_MICRO_BSZ), "試走 subset の micro batch 構成が公式と一致しない"
        print(f"試走 subset の micro batch 構成: 公式 {OFFICIAL_WORLD_SIZE} GPU と一致（{_trial_rows // OFFICIAL_TBS} step 分）")
    print(f"\n[試走 A] step {TRIAL_KILL_AFTER_STEP} の checkpoint 保存完了を検知 → 学習プロセスを一時停止 → HF へ転送・検証 → プロセス終了（切断の模擬）")
    TRIAL_RESULTS["A_interrupted"] = launch_training(TRIAL_SPEC, kill_after_step=TRIAL_KILL_AFTER_STEP, upload=True, poll_sec=5, stable_scans=False)
    _a = TRIAL_RESULTS["A_interrupted"]
    assert _a["killed_by_monitor"], "監視スレッドによる終了が起きていない（プロセスが先に完走した、または HF 転送が完了していない）"
    assert TRIAL_KILL_AFTER_STEP in _a["monitor"]["uploaded"], "kill step の checkpoint が HF に転送・検証されていない"
    _a_last = max(m["step"] for m in _a["metrics"])
    assert _a_last < _total, f"run A が最終 step {_total} まで進んでしまい、中断の模擬になっていない（last step {_a_last}）"
    print(f"run A: step {_a_last} で終了（総 step {_total}）。step {TRIAL_KILL_AFTER_STEP + 1}〜{_a_last} は再実行される")
    print(f"run A の checkpoint（HF）: {_a['monitor']['uploaded'][TRIAL_KILL_AFTER_STEP].get('url')}")
    _hf_before = {r["step"]: r.get("folder_commit") for r in list_hf_checkpoints(_trial_id) if r["complete"]}

    # ローカル checkpoint を消して、HF からの取得だけで再開できることを確かめる
    shutil.rmtree(f"{CKPT_DIR}/{_trial_id}", ignore_errors=True)
    print(f"\n[試走 B] ローカル checkpoint を削除し、HF から step {TRIAL_KILL_AFTER_STEP} を取得して新プロセスで再開（総 epoch {TRIAL_EPOCHS} / 総 step {_total} を維持）")
    TRIAL_RESULTS["B_resumed"] = resume_run_from_hf(_trial_id, step=TRIAL_KILL_AFTER_STEP, upload=True)
    _b = TRIAL_RESULTS["B_resumed"]
    assert _b["exit_code"] == 0, "再開 run が異常終了"
    _b_steps = [m["step"] for m in _b["metrics"]]
    assert _b_steps and _b_steps[0] == TRIAL_KILL_AFTER_STEP + 1 and _b_steps[-1] == _total, f"再開後の step 列が想定外: {_b_steps}"
    _b_log = open(_b["log_path"]).read()
    _hf_after = {r["step"]: r.get("folder_commit") for r in list_hf_checkpoints(_trial_id) if r["complete"]}
    RESUME_CHECKS = dict(
        global_step_continued=(_b_steps[0] == TRIAL_KILL_AFTER_STEP + 1),
        model_loaded=("Loaded model from" in _b_log), optimizer_loaded=("Loaded optimizer from" in _b_log),
        lr_scheduler_loaded=("Loaded lr_scheduler from" in _b_log), rng_loaded=("Loaded rng from" in _b_log),
        dataloader_state_loaded=("Successfully loaded dataloader state" in _b_log),
        final_ckpt_uploaded=(_total in _b["monitor"]["uploaded"] and not _b["monitor"]["uploaded"][_total].get("previously_verified")),
        kill_step_ckpt_not_overwritten=(_hf_before.get(TRIAL_KILL_AFTER_STEP) == _hf_after.get(TRIAL_KILL_AFTER_STEP) is not None),
        no_upload_failures=(not _b["monitor"]["failed"]),
    )
    print("再開チェック:", RESUME_CHECKS)

    # 解析的な cosine schedule と比較（warmup = int(total*0.1)、verl get_cosine_schedule_with_warmup と同式）
    def expected_lr(step, total, base_lr, warmup_ratio=float(YAML_DEFAULTS.optim.warmup_steps_ratio)):
        warm = int(total * warmup_ratio)
        if step < warm:
            return base_lr * step / max(1, warm)
        prog = (step - warm) / max(1, total - warm)
        return base_lr * max(0.0, 0.5 * (1 + math.cos(math.pi * prog)))

    LR_CHECK = []
    for m in _b["metrics"]:
        exp = expected_lr(m["step"], _total, OFFICIAL_BASELINE["lr"])   # train/lr は step 後の get_last_lr
        LR_CHECK.append(dict(step=m["step"], logged=m.get("train/lr"), expected_after_step=exp, ok=abs(m.get("train/lr", -1) - exp) <= 1e-9 + 1e-3 * abs(exp)))
    print("lr の継続（再開後）:", LR_CHECK)
    RESUME_CHECKS["lr_schedule_continued"] = all(x["ok"] for x in LR_CHECK)

    if TRIAL_RUN_REFERENCE:
        print(f"\n[試走 C] 中断なしの参照 run（同じ設定）。再開後の loss / データ位置を比較する")
        _ref_spec = build_run_spec("trial", OFFICIAL_BASELINE["lr"], TRIAL_EPOCHS, data_path=TRIAL_PARQUET, data_sha256=TRIAL_SHA256, n_rows=_trial_rows,
                                   max_length=MAX_LENGTH, save_freq=TRIAL_SAVE_FREQ, upload_policy="all", run_id=_trial_id + "-ref", extra_note="試走の参照 run（中断なし）")
        TRIAL_RESULTS["C_reference"] = launch_training(_ref_spec, upload=False, keep_local=99)
        _c = {m["step"]: m for m in TRIAL_RESULTS["C_reference"]["metrics"]}
        _cmp = []
        for m in _b["metrics"]:
            r = _c.get(m["step"])
            if r:
                _cmp.append(dict(step=m["step"], loss_resumed=m.get("train/loss"), loss_reference=r.get("train/loss"),
                                 rel_diff=abs(m.get("train/loss", 0) - r.get("train/loss", 0)) / max(1e-8, abs(r.get("train/loss", 0))),
                                 lr_equal=abs(m.get("train/lr", 0) - r.get("train/lr", 0)) < 1e-12))
        print("再開 run と参照 run の比較（同じ step で同じバッチなら loss は近い。大きくずれればデータ位置が復元されていない）:")
        for x in _cmp:
            print("  ", x)
        # 参照 run が正常に終わり、再開後の全 step を比べられたときだけ合格にする（参照 run が途中で止まると、一部の step しか確認できていない）
        _ref_ok = TRIAL_RESULTS["C_reference"]["exit_code"] == 0 and len(_cmp) == len(_b["metrics"])
        if not _ref_ok:
            print(f"*** 参照 run と比べられたのは {len(_cmp)} / {len(_b['metrics'])} step（参照 run の終了コード {TRIAL_RESULTS['C_reference']['exit_code']}）。"
                  "全 step を確認できていないので、参照 run との比較 2 項目は不合格にする ***")
        RESUME_CHECKS["loss_matches_reference_within_5pct"] = _ref_ok and all(x["rel_diff"] < 0.05 for x in _cmp)
        RESUME_CHECKS["lr_equals_reference"] = _ref_ok and all(x["lr_equal"] for x in _cmp)

    # ---- 時間計測と本学習の概算 ----
    _tA, _tB = summarize_timing(_a), summarize_timing(_b)
    _tC = summarize_timing(TRIAL_RESULTS["C_reference"]) if TRIAL_RUN_REFERENCE else None
    _step_sec = np.median([x for t in (_tA, _tB, _tC) if t and t.get("step_sec_median") for x in [t["step_sec_median"]]])
    _init_sec = _tA.get("init_sec_est") or _tA.get("init_sec_incl_first_step")
    _save_sec = float(np.median([s for t in (_tA, _tB) for s in t.get("save_sec", [])])) if any(t.get("save_sec") for t in (_tA, _tB)) else None
    _upload_sec = float(np.median([s for t in (_tA, _tB) for s in t.get("upload_sec", {}).values()])) if any(t.get("upload_sec") for t in (_tA, _tB)) else None
    print("\n=== 試走の計測 ===")
    print(f"  初期化（起動〜最初の step 前）≈ {_init_sec:.0f}s | optimizer step（勾配蓄積 {TRIAL_SPEC['grad_accum_micro_batches']} micro batch 込み）中央値 {_step_sec:.1f}s")
    print(f"  checkpoint 保存 {_save_sec}s | HF 転送+検証 {_upload_sec}s（バックグラウンド） | GPU メモリ最大 {_a['monitor']['gpu_mem_max_mb']} MB")
    _base_steps = steps_per_epoch(DATASET_EXPECTED_ROWS, OFFICIAL_TBS, N_GPUS) * OFFICIAL_BASELINE["epochs"]
    _base_saves = len(set(list(range(SAVE_FREQ, _base_steps + 1, SAVE_FREQ)) + [_base_steps]))
    _base_est = _init_sec + _base_steps * _step_sec + _base_saves * (_save_sec or 0)
    print(f"  baseline 概算: init {_init_sec:.0f}s + {_base_steps} steps x {_step_sec:.1f}s + {_base_saves} saves x {_save_sec or 0}s ≈ {_base_est / 3600:.1f} h（HF 転送は並行。評価・変換・最終アップロードは未計測）")
    _grid = [normalize_run(c) for c in (SEARCH_GRID_CONFIRMED or [])]
    if _grid:
        _grid_steps = sum(steps_per_epoch(DATASET_EXPECTED_ROWS, OFFICIAL_TBS, N_GPUS) * e for _, e, _ in _grid)
        print(f"  探索全体（{len(_grid)} 条件 = {_grid_steps} steps）概算 ≈ {(_grid_steps * _step_sec + len(_grid) * _init_sec) / 3600:.1f} h（保存・評価は含まない）")
    print("  注意: 試走は step 上限を使わず epoch 数で短くしたので scheduler 期間は試走内で完結している。試走の checkpoint・重み・データ削減は本学習に使わない")
    TRIAL_TIMING = dict(init_sec=_init_sec, step_sec_median=float(_step_sec), save_sec=_save_sec, upload_sec=_upload_sec, gpu_mem_max_mb=_a["monitor"]["gpu_mem_max_mb"],
                        baseline_est_hours=_base_est / 3600, timing_A=_tA, timing_B=_tB, timing_C=_tC, resume_checks=RESUME_CHECKS, lr_check=LR_CHECK)
    with open(f"{RECORD_DIR}/trial_{_trial_id}.json", "w") as f:
        json.dump(dict(spec=TRIAL_SPEC, timing=TRIAL_TIMING, results={k: dict(exit_code=v["exit_code"], log=v["log_path"], uploaded=list(v["monitor"]["uploaded"])) for k, v in TRIAL_RESULTS.items()}),
                  f, indent=2, ensure_ascii=False, default=str)

    # ---- 合格条件の一覧と HF の URL（この下の行を貼れば、試走の結果を確認できる） ----
    _check_desc = dict(
        global_step_continued="再開後の最初の step が中断した step の次",
        model_loaded="再開時に model を読み込んだ", optimizer_loaded="再開時に optimizer を読み込んだ",
        lr_scheduler_loaded="再開時に lr scheduler を読み込んだ", rng_loaded="再開時に乱数の状態を読み込んだ",
        dataloader_state_loaded="再開時にデータの位置を読み込んだ",
        final_ckpt_uploaded="最終 step の checkpoint を HF へ転送・検証した",
        kill_step_ckpt_not_overwritten="中断した step の checkpoint が再開後も HF で上書きされていない",
        no_upload_failures="HF 転送の失敗なし", lr_schedule_continued="再開後の lr が cosine の式と一致",
        loss_matches_reference_within_5pct="再開 run と中断なしの参照 run の loss の差が 5% 未満", lr_equals_reference="再開 run と参照 run の lr が一致")
    print()
    print_checklist("試走の合格条件", [(_check_desc.get(k, k), v, k) for k, v in RESUME_CHECKS.items()])
    print(f"HF（試走の run）: {hf_run_url(_trial_id)}")
    for _k in ("A_interrupted", "B_resumed"):
        for _s, _v in sorted(TRIAL_RESULTS[_k]["monitor"]["uploaded"].items()):
            if _v.get("url"):
                print(f"  {_k} step {_s}: {_v['url']}")
    if not all(v for v in RESUME_CHECKS.values() if v is not None):
        print("*** 再開チェックに失敗項目がある。原因を解決するまで本学習へ進まない ***")
    else:
        print("試走の再開チェックはすべて合格")
    mirror_record(f"{RECORD_DIR}/trial_{_trial_id}.json", _trial_id)   # 合否を表示した後に写す（写し先が遅くても表示を待たせない）
