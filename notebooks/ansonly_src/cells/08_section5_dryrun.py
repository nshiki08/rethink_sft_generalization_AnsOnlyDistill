# @title 5. ドライラン: データ・設定・コマンド・想定 step 数・保存先を確認する（学習も HF 書き込みも開始しない）
import shutil, json, os


def plan_runs():
    """RUN_KIND に応じて実行予定の (lr, epochs) を返す"""
    if RUN_KIND == "baseline":
        return [(OFFICIAL_BASELINE["lr"], OFFICIAL_BASELINE["epochs"])]
    return list(SEARCH_RUN_LIST)


DRY_RUN_SPECS = []
if not plan_runs():
    print("実行予定の候補が無い（RUN_KIND='search' なら SEARCH_RUN_LIST を設定する）")
for _lr, _ep in plan_runs():
    _spec = build_run_spec(RUN_KIND, _lr, _ep, data_path=AO_PARQUET, data_sha256=AO_SHA256, n_rows=DATASET_EXPECTED_ROWS,
                           max_length=MAX_LENGTH, save_freq=SAVE_FREQ, n_gpus=max(N_GPUS, 1))
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
    _remote = hf_run_paths(HF_CKPT_REPO_ID, _spec["run_id"]) if HF_API else {}
    print(f"  衝突確認: local dir exists={_local_exists} | HF runs/{_spec['run_id']}: {sorted(_remote) if _remote else 'なし'}")
    if _local_exists or _remote:
        print("  *** 同名 run が既に存在する。上書きしない。再開なら『再開』セル、別 run なら RUN_ID_SUFFIX を設定 ***")
    with open(f"{RECORD_DIR}/dry_run_{_spec['run_id']}.json", "w") as f:
        json.dump(dict(spec=_spec, command=_cmd, local_exists=_local_exists, remote_steps=_remote), f, indent=2, ensure_ascii=False)

print("=" * 100)
print("torchrun:", shutil.which("torchrun"), "| GPUs:", N_GPUS, "| AO_DATA_READY:", AO_DATA_READY, "| MASK_CHECK_OK:", MASK_CHECK_OK, "| flash_attn:", FLASH_ATTN_OK)
print("ドライラン完了。学習・HF 書き込みは行っていない。RUN_MODE を 'trial' / 'train' にして先へ進む。")
