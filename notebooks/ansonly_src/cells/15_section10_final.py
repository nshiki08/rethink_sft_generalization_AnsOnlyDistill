# @title 10. 最終モデル: verl.model_merger で HF 形式へ変換 → HF 保存 → 読み込み・短い生成の確認 → Model Card
import json, os, time, torch, glob
from transformers import AutoTokenizer, AutoModelForCausalLM

_final_rid = FINAL_RUN_ID or next(iter(globals().get("TRAIN_RESULTS", {}) or {}), None)
if not _final_rid:
    print("FINAL_RUN_ID が未設定で、このセッションに学習 run も無い。設定セルで FINAL_RUN_ID を指定する")
elif RUN_MODE == "dry_run":
    print("dry_run では変換・アップロードを行わない")
else:
    assert HF_API is not None and HF_CKPT_REPO_ID, "HF にログインしていない"
    _ckpts = list_hf_checkpoints(_final_rid)
    _complete = [c["step"] for c in _ckpts if c["complete"]]
    print(f"{_final_rid}: HF 完了済み steps = {_complete}")
    if FINAL_STEP == "last":
        _tot = _ckpts[-1].get("total_steps") if _ckpts else None
        _step = max(_complete) if _complete else None
        if _tot and _step != _tot:
            print(f"WARN: 最終 step {_tot} の checkpoint が HF に無い（最新は {_step}）。学習が完了していない可能性がある")
    else:
        _step = int(FINAL_STEP)
    assert _step in _complete, f"step {_step} は HF に完了済みでない"
    MERGED_DIR = merge_checkpoint(_final_rid, _step)
    _saved_spec = get_manifest(_final_rid, _step)["run_config"]
    FINAL_REPO = HF_FINAL_MODEL_REPO_ID or f"{HF_ACCOUNT_NAME}/{_final_rid}"
    _sub = f"step{_step}"   # 公開 CoT モデルと同じ stepNNN/ サブディレクトリ構成
    HF_API.create_repo(FINAL_REPO, private=HF_FINAL_MODEL_PRIVATE, repo_type="model", exist_ok=True)
    _existing = [f for f in HF_API.list_repo_files(FINAL_REPO, repo_type="model") if f.startswith(_sub + "/")]
    if _existing:
        raise SystemExit(f"{FINAL_REPO}/{_sub} は既に存在する。上書きしない。HF_FINAL_MODEL_REPO_ID を変える")

    _paper_evals = [r for r in globals().get("PAPER_EVAL_RESULTS", []) if r.get("run_id") == _final_rid]
    if not _paper_evals:   # 別のセッションで評価した場合は HF の runs/<run_id>/paper_eval/ から step ごとに最新の結果を読む
        _latest = {}
        for _f in sorted(f for f in HF_API.list_repo_files(HF_CKPT_REPO_ID, repo_type="model") if f.startswith(f"runs/{_final_rid}/paper_eval/")):
            _latest[_f.split("/")[-1].split("_")[0]] = _f   # step<N>_<時刻>.json → 時刻順で最後が最新
        _paper_evals = [json.load(open(hf_hub_download(HF_CKPT_REPO_ID, _f, repo_type="model", local_dir=f"{WORK_DIR}/hf_markers"))) for _f in _latest.values()]
        if _paper_evals:
            print(f"論文と同じ評価の結果を HF から読んだ: {len(_paper_evals)} step")
    _pe_md = "\n".join(f"| {r['step']} | " + " | ".join(f"{r['results'][d][f'avg@{PAPER_EVAL_K[d]}']:.1f} ({'' if r['results'][d].get('paper_reference') is None else r['results'][d]['paper_reference']})"
                                                         for d in PAPER_EVAL_DATASETS) + f" | {r['results'][PAPER_EVAL_DATASETS[0]]['avg_length_tokens']:.0f} |"
                        for r in sorted(_paper_evals, key=lambda r: r["step"])) or "| (未実施) |" + " - |" * (len(PAPER_EVAL_DATASETS) + 1)
    _changes_md = "\n".join(f"| `{c['key']}` | `{c['official']}` | `{c['new']}` | {c['reason']} |" for c in _saved_spec["changes_vs_official"])
    _audit = AO_AUDIT_SUMMARY
    _tv = _saved_spec.get("training_view") or dict(enabled=False)
    if _tv.get("enabled"):
        _align_md = (f"- 学習は {_saved_spec['n_gpus']} GPU。公式 trainer は 1 GPU 内の micro batch の勾配を和で蓄積し GPU 間は平均するので、そのままでは micro batch の組み合わせと勾配の大きさが公式と変わる。\n"
                     f"- 学習用 file の行を並べ替え（`{os.path.basename(_saved_spec['train_file'])}`, sha256 {_saved_spec['train_file_sha256'][:12]}）、各 step の各 micro batch を公式 {_tv['official_world']} GPU と同じ 4 行にした。\n"
                     f"- `trainer.importance_sampling_mode=adv-only` と `advantage={_tv['advantage']}` で loss を {_tv['advantage']} 倍し、勾配を公式と同じ大きさにした（ログの train/loss も {_tv['advantage']} 倍）。\n")
    elif _saved_spec["n_gpus"] == _saved_spec.get("official_world_size", 8):
        _align_md = f"- 公式と同じ {_saved_spec['n_gpus']} GPU で学習した。\n"
    else:
        _align_md = f"- **合わせ込みなし**: {_saved_spec['n_gpus']} GPU で学習し、micro batch の組み合わせと勾配の大きさが公式と異なる。\n"
    _align_md += "- 残る差（公式コードを変えずには揃えられない）:\n" + "\n".join(f"  - {r}" for r in _saved_spec.get("residual_differences", []))
    MODEL_CARD = f"""---
license: apache-2.0
base_model: {_saved_spec['base_repo']}
tags: [answer-only-distillation, sft, math, rethink-sft-generalization]
---
# {_final_rid} (Answer-only distillation student, step {_step})

Answer-only (AO) 蒸留の学生モデル。公開 CoT 学生 `{_saved_spec['cot_repo']}/{_saved_spec['cot_subfolder']}` と比較するために、
同じ Base・同じ prompt・同じ問題で、教師 `{TEACHER['repo']}` の**最終回答だけ**を target に Full-parameter SFT した。
学習コードは公式実装 (`verl.trainer.fsdp_sft_trainer_ours`) を無変更で使用。重みは `{_sub}/` にある（公開 CoT モデルと同じ配置）。

## 対応関係と revision
| 役割 | repo | revision |
| --- | --- | --- |
| Teacher | {TEACHER['repo']} | {TEACHER['revision']} |
| Base | {_saved_spec['base_repo']} | {_saved_spec['base_revision']} |
| CoT student (比較対象) | {_saved_spec['cot_repo']} (subfolder {_saved_spec['cot_subfolder']}) | {_saved_spec['cot_revision']} |
| 学習データ | {_saved_spec['dataset_repo']} | {_saved_spec['dataset_revision']} (raw sha256 {DATASET_EXPECTED_SHA256[:12]}, AO parquet sha256 {_saved_spec['data_sha256'][:12]}) |
| コード (Fork) | {FORK_REPO_URL} | {_saved_spec['fork_commit']} (参照元 {UPSTREAM_REFERENCE_COMMIT}) |
| tokenizer | Base と同一（checkpoint 内 huggingface/ から復元） | - |

## AO データの作り方と監査
- target = `response`（教師 CoT+回答）の `</think>` 以降の最後の `\\boxed{{...}}` を原文のまま（style: `{_saved_spec['target_style']}`）。CoT・説明文は含めない。参照正解 `answer` は照合のみ。
- 抽出: `verl/utils/reward_score/math_verify_ours.py` の `last_boxed_only_string`、照合: 同 `compute_score`（math-verify）。
- 行数 {_audit['rows']}、抽出状況 {json.dumps(_audit['extraction_status'])}、照合 {json.dumps(_audit['verify_status'])}、
  複数の異なる box を持つ行 {_audit['rows_multi_distinct_boxes']}、照合不一致で目視確認のうえ保持した行 {AO_ACKNOWLEDGED_VERIFY_MISMATCH_ROWS}。
- AO parquet では元の 20,480 行・行順・`message`・`advantage` を維持。抽出失敗時の `answer` 代用や行削除はしない。
- prompt は公式と同一で「step by step」の指示を含むが、target は回答のみ。

## 公式 CoT 学習（{_saved_spec.get('official_world_size', 8)} GPU）との条件合わせ
{_align_md}

## 学習設定
- run_id `{_final_rid}`, lr {_saved_spec['lr_str']}, epochs {_saved_spec['epochs']}, global batch {_saved_spec['tbs']}, micro batch/GPU {_saved_spec['micro_bsz']}, GPU 数 {_saved_spec['n_gpus']},
  steps/epoch {_saved_spec['steps_per_epoch']}, total steps {_saved_spec['total_steps']}, warmup {_saved_spec['warmup_steps']}, max_length {_saved_spec['max_length']} (公式 {_saved_spec['official_max_length']}), save_freq {_saved_spec['save_freq']}
- 公式スクリプト `{os.path.basename(_saved_spec['official_script'])}` との差分:

| key | 公式 | 本 run | 理由 |
| --- | --- | --- | --- |
{_changes_md}

- 環境: {json.dumps(ENV_RECORD.get('gpus'))}, torch {PKG_VERSIONS.get('torch')}, transformers {PKG_VERSIONS.get('transformers')}, flash-attn {FLASH_ATTN_VERSION}, Colab={IN_COLAB}, GPU プロファイル {_saved_spec.get('gpu_profile', 'official')}{('（公式からの変更: ' + _saved_spec['gpu_profile_deviation'] + '）') if _saved_spec.get('gpu_profile_deviation') else ''}
- 再開用 checkpoint（model/optimizer/extra/dataloader 状態）: `{HF_CKPT_REPO_ID}` の `runs/{_final_rid}/global_step_*`

## 評価（論文と同じ方法）
公式 `evaluation/math_eval/math_eval_budget.py` を無変更で実行（vLLM 0.8.5, temperature 0.6, top_p 0.95, 最大 32768 token, math-verify）。
MATH500 は avg@3、AIME24 は avg@10（%）。括弧内は同じ step の公開 CoT 学生の論文値（App. D）。論文との差: {'; '.join(PAPER_EVAL_DEVIATIONS)}。

| step | {' | '.join(PAPER_EVAL_DATASETS)} | {PAPER_EVAL_DATASETS[0]} の平均出力長 (token) |
| --- | {' | '.join('---' for _ in PAPER_EVAL_DATASETS)} | --- |
{_pe_md}

loss の低下だけで能力向上は主張しない。上の表が「未実施」なら性能は未測定。

## 評価の範囲
- 研究目的は数学能力への CoT の寄与を調べることなので、評価は MATH500 / AIME24 だけ。論文の数学以外の評価（LiveCodeBench, GPQA-Diamond, MMLU-Pro, IFEval, AlpacaEval, HaluEval, TruthfulQA, HEx-PHI）は行っていない。
"""
    with open(os.path.join(MERGED_DIR, "..", "README.md"), "w") as f:
        f.write(MODEL_CARD)
    print("uploading merged model to", FINAL_REPO, "/", _sub)
    _info = HF_API.upload_folder(folder_path=MERGED_DIR, path_in_repo=_sub, repo_id=FINAL_REPO, repo_type="model", commit_message=f"{_final_rid} merged step {_step}")
    _info_rd = HF_API.upload_file(path_or_fileobj=os.path.join(MERGED_DIR, "..", "README.md"), path_in_repo="README.md", repo_id=FINAL_REPO, repo_type="model",
                                  commit_message=f"model card for {_final_rid}")
    FINAL_MODEL = dict(repo=FINAL_REPO, subfolder=_sub, revision=_info_rd.oid, weights_commit=_info.oid, url=f"https://huggingface.co/{FINAL_REPO}/tree/{_info_rd.oid}/{_sub}",
                       source_checkpoint=dict(repo=HF_CKPT_REPO_ID, path=f"runs/{_final_rid}/global_step_{_step}"))
    print("uploaded:", FINAL_MODEL)

    # 読み込みと短い生成の確認（アップロード後の revision から）
    _tok = AutoTokenizer.from_pretrained(FINAL_REPO, subfolder=_sub, revision=_info_rd.oid, trust_remote_code=True)
    _mdl = AutoModelForCausalLM.from_pretrained(FINAL_REPO, subfolder=_sub, revision=_info_rd.oid, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    _probe_prompts = [list(RAW_TABLE.column("message")[0].as_py()),
                      [{"role": "user", "content": "What is 17 + 26?\n Please reason step by step, and put your final answer within \\boxed{}."}]]
    GEN_CHECK = []
    for _p in _probe_prompts:
        _enc = _tok.apply_chat_template(_p, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
        with torch.no_grad():
            _g = _mdl.generate(**_enc, max_new_tokens=64, do_sample=False, eos_token_id=_tok.eos_token_id, pad_token_id=_tok.pad_token_id)
        _out = _tok.decode(_g[0, _enc["input_ids"].shape[1]:], skip_special_tokens=False)
        GEN_CHECK.append(dict(prompt=_p[0]["content"][:80], output=_out))
        print("prompt:", _p[0]["content"][:80].replace("\n", " "), "\n  -> output:", repr(_out))
    del _mdl; torch.cuda.empty_cache()
    FINAL_MODEL["generation_check"] = GEN_CHECK
    _own_results = [r for r in globals().get("TRAIN_RESULTS", {}).values() if r["run_id"] == _final_rid]
    write_experiment_record(_saved_spec, _own_results, evaluations=_paper_evals, final_model=FINAL_MODEL,
                            unverified=(["論文と同じ評価（MATH500/AIME24）未実施"] if not _paper_evals else []) +
                                       (["このセッションの学習ログなし（HF の experiment_record を参照）"] if not _own_results else []))
