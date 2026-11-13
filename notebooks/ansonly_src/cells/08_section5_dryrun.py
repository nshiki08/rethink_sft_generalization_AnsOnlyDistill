# @title 5. ドライラン: データ・設定・コマンド・想定 step 数・保存先を確認する（学習も HF 書き込みも開始しない）
import shutil, json, os


def plan_runs():
    """RUN_KIND に応じて実行予定の (lr, epochs, scheduler) を返す"""
    if RUN_KIND == "baseline":
        return [(OFFICIAL_BASELINE["lr"], OFFICIAL_BASELINE["epochs"], "cosine")]
    return [normalize_run(r) for r in SEARCH_RUN_LIST]


DRY_RUN_SPECS = []
_collisions = []   # ローカルか HF に同名 run がある run_id
if not plan_runs():
    print("実行予定の候補が無い（RUN_KIND='search' なら SEARCH_RUN_LIST を設定する）")
for _lr, _ep, _sched in plan_runs():
    _spec = build_run_spec(RUN_KIND, _lr, _ep, data_path=AO_PARQUET, data_sha256=AO_SHA256, n_rows=DATASET_EXPECTED_ROWS,
                           max_length=MAX_LENGTH, save_freq=SAVE_FREQ, n_gpus=max(N_GPUS, 1), lr_scheduler=_sched)
    DRY_RUN_SPECS.append(_spec)
    print("=" * 100)
    print_run_spec(_spec)
    _cmd = ["torchrun", "--nnodes=1", f"--nproc_per_node={_spec['n_gpus']}", "--node_rank=0", "--master_addr=127.0.0.1", "--master_port=<random>",
            "-m", _spec["module"]] + [f"{k}={v}" for k, v in _spec["overrides"].items()]
    print("  command:\n    " + " \\\n    ".join(_cmd))
    print("  env:", _spec["env"])
    _nfull = len(_spec["upload_full_steps"])
    _nlight = len(_spec["upload_steps"]) - _nfull
    print(f"  checkpoint: 保存 {len(_spec['expected_save_steps'])} 回、HF 転送 {len(_spec['upload_steps'])} 回（再開用 {_nfull} x {CKPT_SIZE_EST_GB:.1f} GB + "
          f"重みのみ {_nlight} x {CKPT_MODEL_ONLY_EST_GB:.1f} GB ≈ {hf_usage_gb(_nfull, _nlight):.0f} GB。HF PRO の非公開枠は 1 TB）、"
          f"ローカルは転送・検証済み {LOCAL_KEEP_LAST_N_VERIFIED_CKPTS} 個を保持")
    _local_exists = os.path.exists(_spec["overrides"]["trainer.default_local_dir"])
    _remote = hf_run_paths(resolve_ckpt_repo_id(), _spec["run_id"]) if HF_API else {}
    print(f"  衝突確認: local dir exists={_local_exists} | HF runs/{_spec['run_id']}: {sorted(_remote) if _remote else 'なし'}")
    if _local_exists or _remote:
        _collisions.append(_spec["run_id"])
        print("  *** 同名 run が既に存在する。上書きしない。再開なら『再開』セル、別 run なら RUN_ID_SUFFIX を設定 ***")
    with open(f"{RECORD_DIR}/dry_run_{_spec['run_id']}.json", "w") as f:
        json.dump(dict(spec=_spec, command=_cmd, local_exists=_local_exists, remote_steps=_remote), f, indent=2, ensure_ascii=False)

print("=" * 100)
print("torchrun:", shutil.which("torchrun"), "| GPUs:", N_GPUS, "| AO_DATA_READY:", AO_DATA_READY, "| MASK_CHECK_OK:", MASK_CHECK_OK, "| flash_attn:", FLASH_ATTN_OK)
# 学習を始める前に満たしておく条件を、ここで一覧にする（このセルは NG でも止めない）。
# 本学習のセルは、開始前に「公式コード」以外の行を確かめて止まる。試走のセルが確かめるのは GPU・flash-attn・FSDP2 勾配確認・AO データ・HF。
# 「公式コード」の行は表示と記録だけ（再開時には、保存時と違えば照合する）。
_ckpt_repo = resolve_ckpt_repo_id()
_need_grad_check = REQUIRE_FSDP2_GRAD_CHECK and N_GPUS == 1
DRY_RUN_CHECKS = [
    ("実行予定の run がある", bool(DRY_RUN_SPECS), f"{len(DRY_RUN_SPECS)} 件"),
    ("GPU がある", N_GPUS >= 1, f"{N_GPUS} 台"),
    ("flash-attn が使える", bool(FLASH_ATTN_OK), None),
    ("1 GPU の FSDP2 勾配確認（セクション 2）", (FSDP2_GRAD_CHECK.get("status") == "ok") if _need_grad_check else None, FSDP2_GRAD_CHECK.get("status")),
    ("公式コード（verl/, training_scripts/）が参照 commit と同じ", OFFICIAL_CODE_UNCHANGED is True, None),
    ("AO データ（セクション 3 の AO_DATA_READY）", bool(AO_DATA_READY), None),
    ("loss mask（セクション 4-b の MASK_CHECK_OK）", bool(MASK_CHECK_OK), None),
    (f"公式 {OFFICIAL_WORLD_SIZE} GPU の再現（セクション 4-b の VIEW_CHECK_OK）", bool(VIEW_CHECK_OK) if EMULATE_OFFICIAL_WORLD_SIZE else None,
     f"確認した GPU 台数 {TRAIN_VIEW_CHECK.get('actual_world')}" if EMULATE_OFFICIAL_WORLD_SIZE else "EMULATE_OFFICIAL_WORLD_SIZE=False（公式との差として記録される）"),
    ("HF にログイン済みで checkpoint repo が決まる", HF_API is not None and bool(_ckpt_repo),
     _ckpt_repo if HF_API is not None and _ckpt_repo else ("HF_TOKEN が無い" if HF_API is None else "HF のアカウント名が取れていない")),
    ("同名 run が無い（ローカル・HF）", not _collisions, ", ".join(_collisions) or None),
]
_dry_run_ng = print_checklist("ドライランの合格条件（すべて OK なら試走へ進める）", DRY_RUN_CHECKS)
print("ドライランの合格条件:", "すべて OK" if not _dry_run_ng else f"NG {len(_dry_run_ng)} 件 → {_dry_run_ng}")
_mirror_dir = log_mirror_dir()
_mirror_problem = mirror_dir_problem(_mirror_dir) if _mirror_dir else None
print("学習ログの写し先:", (_mirror_dir + (f" *** 写せない: {_mirror_problem} ***" if _mirror_problem else "")) if _mirror_dir
      else "なし（切断すると loss の推移とログが消える。Colab Secrets に AO_DRIVE_LOG_DIR を登録してセル 0 を実行し直す）")
print("ドライラン完了。学習・HF 書き込みは行っていない。RUN_MODE を 'trial' / 'train' にして先へ進む。")
