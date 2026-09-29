# @title 9. dev 評価と結果集計（既存の評価 parser + math-verify を流用。dev 未確定なら選択処理の準備に留める）
import importlib.util, json, os, sys, time, torch, gc
from math_verify import parse as mv_parse, verify as mv_verify
from transformers import AutoTokenizer, AutoModelForCausalLM

# 既存の評価コード evaluation/math_eval/utils/parser.py の extract_answer を読む（math_eval_budget.py と同じ抽出）
_ME_DIR = f"{REPO_DIR}/evaluation/math_eval"
if _ME_DIR not in sys.path:
    sys.path.insert(0, _ME_DIR)
_spec_p = importlib.util.spec_from_file_location("math_eval_parser", f"{_ME_DIR}/utils/parser.py")
_parser_mod = importlib.util.module_from_spec(_spec_p)
_spec_p.loader.exec_module(_parser_mod)
extract_answer = _parser_mod.extract_answer


def build_eval_prompt(question):
    """公式評価 (evaluation/math_eval/math_eval_budget.py build_prompt) と同じ"""
    return [{"role": "user", "content": "{}\n Please reason step by step, and put your final answer within \\boxed{{}}.".format(question)}]


def load_dev_rows():
    if not DEV_EVAL_SOURCE:
        return None
    rows = []
    if os.path.isfile(DEV_EVAL_SOURCE):
        with open(DEV_EVAL_SOURCE) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        from datasets import load_dataset
        ds = load_dataset(DEV_EVAL_SOURCE, split=DEV_EVAL_SPLIT, revision=DEV_EVAL_REVISION)
        rows = [dict(r) for r in ds]
    out = [dict(question=r[DEV_EVAL_QUESTION_KEY], answer=(r[DEV_EVAL_ANSWER_KEY][0] if isinstance(r[DEV_EVAL_ANSWER_KEY], list) else r[DEV_EVAL_ANSWER_KEY])) for r in rows]
    return out[:DEV_EVAL_MAX_ROWS] if DEV_EVAL_MAX_ROWS else out


def score_responses(rows, responses):
    """math_eval_budget.py の process_single_result_no_truncation と同じ照合（gold を \\boxed{} で包み、応答全体を parse して verify）"""
    per = []
    for r, resp in zip(rows, responses):
        gold = str(r["answer"])
        gold_parsed = mv_parse(gold) if "boxed" in gold else mv_parse("\\boxed{" + gold + "}")
        try:
            pred = mv_parse(resp)
        except Exception:  # noqa
            pred = resp
        try:
            ok = bool(mv_verify(gold_parsed, pred))
        except Exception:  # noqa
            ok = False
        per.append(dict(gold=gold, extracted=extract_answer(resp, data_name="math", use_last_number=False), correct=ok, n_chars=len(resp)))
    acc = sum(p["correct"] for p in per) / max(1, len(per))
    return acc, per


def generate_hf(path_or_repo, prompts, subfolder=None, revision=None, max_new_tokens=512, batch_size=16, do_sample=False):
    kw = dict(trust_remote_code=True)
    if subfolder:      # ローカルディレクトリに subfolder=None を渡すと transformers 4.52 は失敗するので、指定時だけ渡す
        kw["subfolder"] = subfolder
    if revision:
        kw["revision"] = revision
    tok = AutoTokenizer.from_pretrained(path_or_repo, **kw)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path_or_repo, torch_dtype=torch.bfloat16, **kw).cuda().eval()
    texts = [tok.apply_chat_template(p, add_generation_prompt=True, tokenize=False) for p in prompts]
    outs = []
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    if do_sample:
        gen_kwargs.update(temperature=0.6, top_p=0.95)   # 公式評価のサンプリング設定
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            gen = model.generate(**enc, **gen_kwargs)
            outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    del model
    gc.collect(); torch.cuda.empty_cache()
    return outs


def evaluate_target(target, rows):
    t0 = time.time()
    prompts = [build_eval_prompt(r["question"]) for r in rows]
    resps = generate_hf(target["path_or_repo"], prompts, subfolder=target.get("subfolder"), revision=target.get("revision"),
                        max_new_tokens=target.get("max_new_tokens", DEV_EVAL_MAX_NEW_TOKENS), batch_size=DEV_EVAL_BATCH_SIZE, do_sample=DEV_EVAL_DO_SAMPLE)
    acc, per = score_responses(rows, resps)
    res = dict(name=target["name"], path_or_repo=target["path_or_repo"], subfolder=target.get("subfolder"), revision=target.get("revision"),
               dev_source=DEV_EVAL_SOURCE, dev_split=DEV_EVAL_SPLIT, dev_revision=DEV_EVAL_REVISION, n=len(rows), accuracy=acc, do_sample=DEV_EVAL_DO_SAMPLE,
               max_new_tokens=target.get("max_new_tokens", DEV_EVAL_MAX_NEW_TOKENS), elapsed_sec=time.time() - t0, scorer="math_eval_budget.py と同じ math-verify 照合 (greedy 1 サンプル)")
    os.makedirs(f"{RECORD_DIR}/dev_eval", exist_ok=True)
    with open(f"{RECORD_DIR}/dev_eval/{target['name'].replace('/', '_')}.jsonl", "w") as f:
        for r, resp, p in zip(rows, resps, per):
            f.write(json.dumps(dict(question=r["question"], response=resp, **p), ensure_ascii=False) + "\n")
    print(f"  {target['name']}: acc={acc:.4f} (n={len(rows)}, {time.time() - t0:.0f}s)")
    return res


DEV_EVAL_RESULTS = globals().get("DEV_EVAL_RESULTS", [])
DEV_ROWS = load_dev_rows() if DEV_EVAL_ENABLED else None
if not DEV_EVAL_ENABLED or not DEV_ROWS:
    print("dev は未確定（DEV_EVAL_ENABLED=False または DEV_EVAL_SOURCE 未設定）。評価と最良設定の選択は行わない。")
    print("準備済み: build_eval_prompt / generate_hf / score_responses / evaluate_target。dev を決めたら DEV_EVAL_* を設定して再実行する。")
    print("注意: 学習データ (Math-CoT-20k) や最終 test をここに指定しない。公開 20,480 行を train/dev に分割しない。")
else:
    assert N_GPUS >= 1
    assert DEV_EVAL_IS_INDEPENDENT, "DEV_EVAL_IS_INDEPENDENT=True を設定する前に、dev が学習データ・最終 test と独立であることを確認する"
    _forbidden = {DATASET_REPO, DATASET_ORIGIN_POOL, os.path.abspath(AO_PARQUET), os.path.abspath(RAW_PARQUET)}
    assert DEV_EVAL_SOURCE not in _forbidden and (not os.path.isfile(DEV_EVAL_SOURCE) or os.path.abspath(DEV_EVAL_SOURCE) not in _forbidden), \
        "学習データ (Math-CoT-20k / OpenR1-Math-220k / AO parquet) を dev にしない"
    print(f"dev: {DEV_EVAL_SOURCE} split={DEV_EVAL_SPLIT} rows={len(DEV_ROWS)}")
    _targets = list(DEV_EVAL_TARGETS)
    if not _targets:
        # このセッションの run の最終 step を候補にする（FSDP checkpoint は verl.model_merger で変換してから評価）
        for _rid, _res in globals().get("TRAIN_RESULTS", {}).items():
            if _res["exit_code"] == 0:
                _st = _res["spec"]["total_steps"]
                _targets.append(dict(name=f"{_rid}@step{_st}", path_or_repo=merge_checkpoint(_rid, _st), subfolder=None, revision=None))
    for _t in _targets:
        DEV_EVAL_RESULTS.append(evaluate_target(_t, DEV_ROWS))
    with open(f"{RECORD_DIR}/dev_eval_results.json", "w") as f:
        json.dump(DEV_EVAL_RESULTS, f, indent=2, ensure_ascii=False)
    print("\n=== dev 集計 ===")
    for _r in sorted(DEV_EVAL_RESULTS, key=lambda x: -x["accuracy"]):
        print(f"  {_r['accuracy']:.4f}  n={_r['n']}  {_r['name']}")
    _search_res = [r for r in DEV_EVAL_RESULTS if "_search" in r["name"] or "_baseline" in r["name"]]
    if _search_res:
        _best = max(_search_res, key=lambda x: x["accuracy"])
        print(f"dev 上の最良: {_best['name']} (acc {_best['accuracy']:.4f})。候補全体を評価し終えるまで『確定』としない。")
