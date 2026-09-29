# @title 4-b. 長さ・target・loss mask・実効設定の確認（公式 tokenizer / chat template / OurSFTDataset をそのまま使う）
import numpy as np, torch, json, time, math, os, glob, collections
from omegaconf import OmegaConf
from verl.utils import hf_tokenizer
from verl.utils.dataset.sft_dataset import OurSFTDataset

BASE_MODEL_LOCAL = get_base_model_local()
print("base model local:", BASE_MODEL_LOCAL)
TOKENIZER = hf_tokenizer(BASE_MODEL_LOCAL, trust_remote_code=hydra_value(OFFICIAL["overrides"].get("model.trust_remote_code", False)))
# 公式 run_sft と同じ eos 上書き処理（公式スクリプトは data.eos_token=null / eos_token_id=null）
if hydra_value(OFFICIAL["overrides"].get("data.eos_token")) is not None:
    TOKENIZER.eos_token = hydra_value(OFFICIAL["overrides"]["data.eos_token"])
if hydra_value(OFFICIAL["overrides"].get("data.eos_token_id")) is not None:
    TOKENIZER.eos_token_id = hydra_value(OFFICIAL["overrides"]["data.eos_token_id"])
print(f"tokenizer: eos={TOKENIZER.eos_token!r}({TOKENIZER.eos_token_id}) pad={TOKENIZER.pad_token!r}({TOKENIZER.pad_token_id}) bos={TOKENIZER.bos_token!r}")

# ---- 4.1 全行の Q + 回答 + EOS 長（OurSFTDataset.__getitem__ と同じ手順） ----
_msgs = RAW_TABLE.column("message").to_pylist()
_targets_all = AO_TABLE.column("teacher_answer").to_pylist()
t0 = time.time()
PROMPT_LENS = np.array([len(TOKENIZER.apply_chat_template(list(m), add_generation_prompt=True, tokenize=True)) for m in _msgs])
RESP_LENS = np.array([len(TOKENIZER(t if t is not None else "", add_special_tokens=False)["input_ids"]) + 1 for t in _targets_all])  # +1 = EOS
TOTAL_LENS = PROMPT_LENS + RESP_LENS
print(f"tokenized {len(TOTAL_LENS)} rows in {time.time() - t0:.1f}s")
for _name, _arr in (("prompt", PROMPT_LENS), ("target+EOS", RESP_LENS), ("total", TOTAL_LENS)):
    print(f"  {_name:11s} min={_arr.min()} mean={_arr.mean():.1f} p50={int(np.median(_arr))} p95={int(np.percentile(_arr, 95))} p99={int(np.percentile(_arr, 99))} max={_arr.max()}")
_example_prompt = TOKENIZER.apply_chat_template(list(_msgs[0]), add_generation_prompt=True, tokenize=False)
print("chat template 例 (row 0):", repr(_example_prompt[:120]), "...", repr(_example_prompt[-40:]))
print("target 例 (row 0):", repr(_targets_all[0]), "-> tokens", TOKENIZER.convert_ids_to_tokens(TOKENIZER(_targets_all[0], add_special_tokens=False)["input_ids"]), "+ EOS")

MAX_TOTAL_LEN = int(TOTAL_LENS.max())
AUTO_FIT_MAX_LENGTH = int(math.ceil(MAX_TOTAL_LEN / MAX_LENGTH_ROUND_TO) * MAX_LENGTH_ROUND_TO)
if MAX_LENGTH_MODE == "auto_fit":
    MAX_LENGTH = AUTO_FIT_MAX_LENGTH
elif MAX_LENGTH_MODE == "official":
    MAX_LENGTH = OFFICIAL_MAX_LENGTH
else:
    MAX_LENGTH = int(MAX_LENGTH_MODE)
_n_trunc = int((TOTAL_LENS > MAX_LENGTH).sum())
print(f"\n全行が切り詰められずに収まる最大長 = {MAX_TOTAL_LEN} tokens（{MAX_LENGTH_ROUND_TO} の倍数へ切り上げ: {AUTO_FIT_MAX_LENGTH}）")
print(f"data.max_length = {MAX_LENGTH}（mode={MAX_LENGTH_MODE}、公式 {OFFICIAL_MAX_LENGTH}）。この長さで切り詰められる行数: {_n_trunc}")
print(f"公式 20000 での padding 率: {(1 - TOTAL_LENS.mean() / OFFICIAL_MAX_LENGTH) * 100:.1f}% / 選択値での padding 率: {(1 - TOTAL_LENS.mean() / MAX_LENGTH) * 100:.1f}%")
if _n_trunc > 0:
    print("*** 切り詰められる行がある。data.truncation=right で target/EOS が失われる → max_length を上げる ***")

# ---- 4.2 公式 OurSFTDataset で loss mask を確認 ----
_data_cfg = OmegaConf.create(OmegaConf.to_container(YAML_DEFAULTS.data, resolve=True))
for _k, _v in OFFICIAL["overrides"].items():
    if _k.startswith("data."):
        OmegaConf.update(_data_cfg, _k[len("data."):], hydra_value(_v), merge=True)
OmegaConf.update(_data_cfg, "train_files", AO_PARQUET)
OmegaConf.update(_data_cfg, "val_files", AO_PARQUET)
OmegaConf.update(_data_cfg, "response_key", "teacher_answer")
OmegaConf.update(_data_cfg, "max_length", int(MAX_LENGTH))
print("\ndataset config (data.*):", json.dumps(OmegaConf.to_container(_data_cfg), ensure_ascii=False)[:600])
DS = OurSFTDataset(parquet_files=AO_PARQUET, tokenizer=TOKENIZER, config=_data_cfg)
assert len(DS) == DATASET_EXPECTED_ROWS


def check_item(i):
    it = DS[i]
    p, r = int(PROMPT_LENS[i]), int(RESP_LENS[i])
    ids, am, lm = it["input_ids"], it["attention_mask"], it["loss_mask"]
    ok = dict(
        length=(ids.shape[0] == MAX_LENGTH),
        no_truncation=(int(am.sum()) == p + r),
        loss_mask_count=(int(lm.sum()) == r),                                              # target tokens + EOS の数
        loss_mask_positions=(torch.nonzero(lm).flatten().tolist() == list(range(p - 1, p + r - 1))),  # 1 つ前の位置が次 token を予測
        prompt_masked=(int(lm[: p - 1].sum()) == 0),
        padding_masked=(int(lm[p + r - 1:].sum()) == 0 and int(am[p + r:].sum()) == 0),
        eos_supervised=(int(ids[p + r - 1]) == TOKENIZER.eos_token_id and int(lm[p + r - 2]) == 1),
        target_tokens=(ids[p: p + r - 1].tolist() == TOKENIZER(_targets_all[i], add_special_tokens=False)["input_ids"]),
        advantage=(float(it["advantages"][0]) == 1.0),
    )
    return ok, it, p, r


t0 = time.time()
_bad = collections.Counter()
FULL_MASK_SCAN = True
_null_rows = [i for i, t in enumerate(_targets_all) if t is None]   # 抽出失敗行（OurSFTDataset は str を要求するので走査から除く。ゲートは AO_DATA_READY）
_scan_rows = [i for i in (range(len(DS)) if FULL_MASK_SCAN else range(0, len(DS), 97)) if _targets_all[i] is not None]
for _i in _scan_rows:
    _ok, _, _, _ = check_item(_i)
    for _k, _v in _ok.items():
        if not _v:
            _bad[_k] += 1
print(f"\nloss mask scan over {len(_scan_rows)} rows in {time.time() - t0:.1f}s -> failures: {dict(_bad) or 'none'} | skipped null-target rows: {len(_null_rows)}")
MASK_CHECK_OK = (len(_bad) == 0) and (len(_null_rows) == 0)
for _i in [i for i in (0, int(TOTAL_LENS.argmax())) if _targets_all[i] is not None]:
    _ok, _it, _p, _r = check_item(_i)
    _sup = _it["input_ids"][_p: _p + _r]
    print(f"row {_i}: prompt_len={_p} target+eos={_r} supervised tokens -> {TOKENIZER.decode(_sup)!r}  checks={_ok}")

# ---- 4.3 実効設定 ----
_param_bytes = sum(os.path.getsize(f) for f in glob.glob(f"{BASE_MODEL_LOCAL}/*.safetensors"))
CKPT_SIZE_EST_GB = _param_bytes * 3 / 1e9   # bf16 params + AdamW の 2 状態（公式は model_dtype=bf16 で読み込む）。1.7B の実測 10.34 GB と 0.2% 以内
CKPT_MODEL_ONLY_EST_GB = _param_bytes / 1e9 + 0.02   # 重み + extra/data.pt/huggingface/（optimizer 状態なし）


def hf_usage_gb(n_full, n_model_only):
    return n_full * CKPT_SIZE_EST_GB + n_model_only * CKPT_MODEL_ONLY_EST_GB


print("\n=== 実効設定（1 GPU あたり micro batch、勾配蓄積） ===")
for _lr, _ep, _kind in [(OFFICIAL_BASELINE["lr"], OFFICIAL_BASELINE["epochs"], "baseline")] + [(a, b, "search") for a, b in SEARCH_RUN_LIST]:
    _spe = steps_per_epoch(DATASET_EXPECTED_ROWS, OFFICIAL_TBS, max(N_GPUS, 1))
    _micro = MICRO_BATCH_OVERRIDE or OFFICIAL_MICRO_BSZ
    _total = int(math.ceil(_spe * _ep))
    _saves = sorted(set(list(range(SAVE_FREQ, _total + 1, SAVE_FREQ)) + [_total]))
    _up = _saves if HF_UPLOAD_STEPS == "all" else [x for x in _saves if x in COT_PUBLIC_STEPS or x % RESUME_CKPT_EVERY == 0 or x == _total]
    _nfull = len(_up) if (HF_UPLOAD_STEPS == "all" or HF_ANALYSIS_STEP_CONTENT == "full") else len([x for x in _up if x % RESUME_CKPT_EVERY == 0 or x == _total])
    print(f"  {_kind:8s} lr={fmt_lr(_lr)} ep={_ep}: steps/epoch={_spe} total={_total} warmup={int(_total * float(YAML_DEFAULTS.optim.warmup_steps_ratio))} "
          f"micro/gpu={_micro} accum={(OFFICIAL_TBS // max(N_GPUS, 1)) // _micro} HF uploads={len(_up)}（再開用 {_nfull} x {CKPT_SIZE_EST_GB:.1f} GB + "
          f"重みのみ {len(_up) - _nfull} x {CKPT_MODEL_ONLY_EST_GB:.1f} GB ≈ {hf_usage_gb(_nfull, len(_up) - _nfull):.0f} GB on HF）")
print("  loss: micro batch ごとの token 平均で backward し、1 GPU 内では micro batch の勾配を「和」で蓄積、GPU 間は FSDP2 が「平均」する（公式 training_step）。"
      "ログの train/loss は micro batch の loss の平均")
print("  optimizer: AdamW betas=%s wd=%s, scheduler=%s warmup_ratio=%s, clip_grad=%s, bf16, FSDP(%s), grad ckpt=%s, shuffle=%s, truncation=%s"
      % (OFFICIAL["overrides"].get("optim.betas"), YAML_DEFAULTS.optim.weight_decay, OFFICIAL["overrides"].get("optim.lr_scheduler"),
         YAML_DEFAULTS.optim.warmup_steps_ratio, YAML_DEFAULTS.optim.clip_grad, YAML_DEFAULTS.model.strategy, YAML_DEFAULTS.model.enable_gradient_checkpointing,
         OFFICIAL["overrides"].get("data.shuffle_train"), OFFICIAL["overrides"].get("data.truncation")))
# ---- 4.4 公式 8 GPU の micro batch 構成と勾配の大きさの再現を確認 ----
_n_check = max(N_GPUS, 1)
if EMULATE_OFFICIAL_WORLD_SIZE and OFFICIAL_WORLD_SIZE % _n_check != 0:
    _vpath, _vsha, TRAIN_VIEW_CHECK = None, None, dict(enabled=False, actual_world=_n_check)
else:
    _vpath, _vsha, TRAIN_VIEW_CHECK = prepare_train_file(AO_PARQUET, AO_SHA256, _n_check)
if EMULATE_OFFICIAL_WORLD_SIZE and OFFICIAL_WORLD_SIZE % _n_check != 0:
    VIEW_CHECK_OK = False
    print(f"\n*** GPU {_n_check} 台は {OFFICIAL_WORLD_SIZE} の約数でないため、公式 {OFFICIAL_WORLD_SIZE} GPU の micro batch 構成を再現できない。1, 2, 4, 8 台のランタイムで実行する ***")
elif _n_check == OFFICIAL_WORLD_SIZE:
    VIEW_CHECK_OK = True
    print(f"\nGPU {_n_check} 台 = 公式と同じ台数。並べ替えと loss 倍率は使わない")
elif not TRAIN_VIEW_CHECK["enabled"]:
    VIEW_CHECK_OK = False
    print(f"\n*** EMULATE_OFFICIAL_WORLD_SIZE=False: GPU {_n_check} 台では micro batch 構成と勾配の大きさ（{OFFICIAL_WORLD_SIZE / _n_check:g} 倍）が公式と異なる ***")
else:
    t0 = time.time()
    _src_rows_view = pq.read_table(_vpath, columns=["ao_source_row"]).column("ao_source_row").to_pylist()
    _official_mb = official_micro_batches(list(range(DATASET_EXPECTED_ROWS)), OFFICIAL_WORLD_SIZE, OFFICIAL_TBS, OFFICIAL_MICRO_BSZ)
    _mine_mb = official_micro_batches(_src_rows_view, _n_check, OFFICIAL_TBS, OFFICIAL_MICRO_BSZ)
    _naive_mb = official_micro_batches(list(range(DATASET_EXPECTED_ROWS)), _n_check, OFFICIAL_TBS, OFFICIAL_MICRO_BSZ)
    _same_steps = sum(a == b for a, b in zip(_official_mb, _mine_mb))
    _naive_same = sum(a == b for a, b in zip(_official_mb, _naive_mb))
    # 公式 dataset クラスで、並べ替え後の file の各行が元の行と同じ tensor（advantage 以外）になることを確認
    DSV = OurSFTDataset(parquet_files=_vpath, tokenizer=TOKENIZER, config=_data_cfg)
    _order = emulate_official_order(DATASET_EXPECTED_ROWS, OFFICIAL_WORLD_SIZE, _n_check, OFFICIAL_TBS, OFFICIAL_MICRO_BSZ)
    _tensor_bad = 0
    for _p in list(range(0, 512)) + list(range(DATASET_EXPECTED_ROWS - 256, DATASET_EXPECTED_ROWS)):   # 最初の 2 step と最後の 1 step
        _a, _b = DSV[_p], DS[_order[_p]]
        if not all(torch.equal(_a[k], _b[k]) for k in ("input_ids", "attention_mask", "position_ids", "loss_mask")):
            _tensor_bad += 1
        if float(_a["advantages"][0]) != TRAIN_VIEW_CHECK["advantage"]:
            _tensor_bad += 1
    del DSV
    VIEW_CHECK_OK = (_same_steps == len(_official_mb) == len(_mine_mb)) and _tensor_bad == 0
    print(f"\n=== 公式 {OFFICIAL_WORLD_SIZE} GPU の再現（GPU {_n_check} 台）: {'OK' if VIEW_CHECK_OK else 'NG'} ({time.time() - t0:.1f}s) ===")
    print(f"  学習用 file: {_vpath} (sha256 {_vsha[:12]})")
    print(f"  各 step の micro batch（4 行の組）が公式と一致した step: {_same_steps}/{len(_official_mb)}（並べ替えない場合: {_naive_same}/{len(_official_mb)}）")
    print(f"  公式 dataset クラスで作った tensor の一致（advantage 以外）と advantage={TRAIN_VIEW_CHECK['advantage']}: 不一致 {_tensor_bad} 行")
    print(f"  trainer.importance_sampling_mode=adv-only: loss = -{TRAIN_VIEW_CHECK['advantage']} × log_prob の token 平均。"
          f"{_n_check} GPU の勾配（micro batch 64 個の和 ÷ {_n_check}）× {TRAIN_VIEW_CHECK['advantage']} = 公式（64 個の和 ÷ {OFFICIAL_WORLD_SIZE}）")

LENGTH_RECORD = dict(max_total_len=MAX_TOTAL_LEN, auto_fit_max_length=AUTO_FIT_MAX_LENGTH, chosen_max_length=int(MAX_LENGTH), official_max_length=OFFICIAL_MAX_LENGTH,
                     rows_truncated_at_chosen=_n_trunc, prompt=dict(min=int(PROMPT_LENS.min()), mean=float(PROMPT_LENS.mean()), max=int(PROMPT_LENS.max())),
                     target_eos=dict(min=int(RESP_LENS.min()), mean=float(RESP_LENS.mean()), max=int(RESP_LENS.max())), mask_check_ok=MASK_CHECK_OK,
                     mask_scan_rows=len(_scan_rows), mask_failures=dict(_bad), ckpt_size_est_gb=CKPT_SIZE_EST_GB, tokenizer_eos=TOKENIZER.eos_token, tokenizer_pad=TOKENIZER.pad_token,
                     official_world_emulation=dict(n_gpus_checked=_n_check, view=TRAIN_VIEW_CHECK, view_check_ok=VIEW_CHECK_OK))
with open(f"{RECORD_DIR}/length_and_mask_record.json", "w") as f:
    json.dump(LENGTH_RECORD, f, indent=2, ensure_ascii=False)
del DS
