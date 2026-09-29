# @title 9. 評価: 9-a 論文と同じ評価（公式 math_eval_budget.py を無変更で実行）/ 9-b 任意の dev 評価（論文には無い）
import json, os, sys, time, glob, shutil, subprocess, hashlib

EVAL_ENV_DIR = ENV_BOOTSTRAP["eval_env_dir"]
EVAL_PY = f"{EVAL_ENV_DIR}/bin/python"
EVAL_DIR = f"{WORK_DIR}/eval"
MATH_EVAL_DIR = f"{REPO_DIR}/evaluation/math_eval"
# 公式の評価スクリプト（math_eval_budget.py の DATASET_CONFIGS、utils/utils.py の DATASET_KEYS）はデータの場所を作者の環境の絶対パスで持つ。
# コードは変えず、このパスに Fork の clone へのシンボリックリンクを置く
OFFICIAL_EVAL_ROOT = "/mnt/shared-storage-user/renqihan/sft_generalization"
PAPER_EVAL_DATASET_PATHS = {"MATH500": f"{OFFICIAL_EVAL_ROOT}/evaluation/data/math/MATH500",
                            "AIME24": f"{OFFICIAL_EVAL_ROOT}/evaluation/data/math/converted_aime_dataset"}
PAPER_EVAL_OUTPUT_KEYS = {"MATH500": ("math500", 2000), "AIME24": ("aime", 5000)}   # (save_name, budget_list[-1])。結果ファイルの場所
PAPER_EVAL_K = {"MATH500": 3, "AIME24": 10}                                         # DATASET_CONFIGS の test_n（論文 App. B.4 と同じ）
PAPER_EVAL_DEVIATIONS = ["tensor_parallel_size 1（論文と公式スクリプトは 2 GPU で 2）。生成の数値は丸め程度に変わり得る",
                         "GPU の種類（論文は H200）"]


def ensure_eval_env():
    """論文の数学評価用の環境（Python 3.12 + vLLM 0.8.5 + 公式 pin）を作る。学習用の公式環境とは別"""
    if GPU_PROFILE != "official":
        raise SystemExit(f"GPU プロファイル {GPU_PROFILE}: vLLM 0.8.5（公式 pin）はこの GPU 用のコードを持たない（sm_80〜sm_90 のみ）。"
                         "評価は A100 のランタイムで行う（学習済みの checkpoint は HF にある）")
    pins = ENV_BOOTSTRAP["eval_pip_pinned"]
    h = hashlib.sha256(json.dumps(pins).encode()).hexdigest()[:16]
    marker = f"{EVAL_ENV_DIR}/ao_eval_env_ok.json"
    if os.path.isfile(marker) and json.load(open(marker)).get("hash") == h and os.path.isfile(EVAL_PY):
        return json.load(open(marker))
    uv_env = {k: v for k, v in os.environ.items() if not k.startswith("UV_") and k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    uv_env.update(UV_CACHE_DIR=ENV_BOOTSTRAP["uv_cache_dir"], UV_PYTHON_INSTALL_DIR=ENV_BOOTSTRAP["uv_python_install_dir"])
    uv = ENV_BOOTSTRAP["uv_cmd"]
    t0 = time.time()
    shutil.rmtree(EVAL_ENV_DIR, ignore_errors=True)
    for cmd in (uv + ["venv", "--quiet", "--python", "3.12", "--python-preference", "only-managed", EVAL_ENV_DIR],
                uv + ["pip", "install", "--quiet", "--python", EVAL_PY] + pins):
        r = subprocess.run(cmd, capture_output=True, text=True, env=uv_env)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:])
            raise RuntimeError("評価用の環境の作成に失敗")
    r = subprocess.run([EVAL_PY, "-c", "import json, vllm, torch, transformers, math_verify, xformers, matplotlib, pandas, datasets; "
                                       "print(json.dumps(dict(vllm=vllm.__version__, torch=torch.__version__, transformers=transformers.__version__, "
                                       "xformers=xformers.__version__, datasets=datasets.__version__)))"],
                       capture_output=True, text=True, env=dict(uv_env, PYTHONPATH=""))
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise RuntimeError("評価用の環境の import 確認に失敗")
    info = dict(hash=h, pins=pins, versions=json.loads(r.stdout.strip().splitlines()[-1]), created_sec=round(time.time() - t0))
    json.dump(info, open(marker, "w"), indent=2)
    print(f"評価用の環境を作成した: {EVAL_ENV_DIR}（{info['created_sec']} 秒）{info['versions']}")
    return info


def ensure_official_eval_paths():
    """公式スクリプトが参照する絶対パスに Fork の clone へのリンクを置き、データがあることを確認する"""
    os.makedirs(os.path.dirname(OFFICIAL_EVAL_ROOT), exist_ok=True)
    if os.path.islink(OFFICIAL_EVAL_ROOT):
        if os.path.realpath(OFFICIAL_EVAL_ROOT) != os.path.realpath(REPO_DIR):
            os.remove(OFFICIAL_EVAL_ROOT)
            os.symlink(REPO_DIR, OFFICIAL_EVAL_ROOT)
    elif os.path.exists(OFFICIAL_EVAL_ROOT):
        raise SystemExit(f"{OFFICIAL_EVAL_ROOT} がリンク以外で存在する。消さずに止める")
    else:
        os.symlink(REPO_DIR, OFFICIAL_EVAL_ROOT)
    for ds, p in PAPER_EVAL_DATASET_PATHS.items():
        assert os.path.isdir(p), f"{ds} のデータが無い: {p}"


def read_official_math_result(out_dir, dataset):
    """math_eval_budget.py が書く results/<name>/<save_name>_budget<B>/no_instruct_512.json の native（打ち切りなし）を読む。
    論文の値 = native の average_pass_rate（MATH500 は avg@3、AIME24 は avg@10）"""
    save_name, budget = PAPER_EVAL_OUTPUT_KEYS[dataset]
    hits = glob.glob(f"{out_dir}/results/*/{save_name}_budget{budget}/no_instruct_512.json")
    if len(hits) != 1:
        return None
    native = json.load(open(hits[0]))["native"]
    return {f"avg@{PAPER_EVAL_K[dataset]}": 100 * float(native["average_pass_rate"]), "pass@1(first sample)": 100 * float(native["pass@1"]),
            "maj@k": 100 * float(native["pass@k(majority)"]), "avg_length_tokens": float(native["average_length"]), "result_file": hits[0]}


def run_official_math_eval(model_dir, dataset, out_dir, log_path):
    """python math_eval_budget.py --model_path ... --dataset ... --types=no_instruct --output_dir ... --num_gpus=1（evaluation/scripts/run_math.sh と同じ引数。GPU 数だけ 1）"""
    cached = read_official_math_result(out_dir, dataset)
    if cached:
        return cached
    cmd = [EVAL_PY, "math_eval_budget.py", f"--model_path={model_dir}", f"--dataset={PAPER_EVAL_DATASET_PATHS[dataset]}", "--types=no_instruct",
           f"--output_dir={out_dir}", "--num_gpus=1"]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    env.update(VLLM_ATTENTION_BACKEND="XFORMERS", VLLM_WORKER_MULTIPROC_METHOD="spawn", CUDA_VISIBLE_DEVICES="0",   # run_math.sh と同じ
               VIRTUAL_ENV=EVAL_ENV_DIR, PATH=f"{EVAL_ENV_DIR}/bin" + os.pathsep + os.environ.get("PATH", ""))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0, last = time.time(), time.time()
    with open(log_path, "w") as lf:
        lf.write(" ".join(cmd) + "\n")
        p = subprocess.Popen(cmd, cwd=MATH_EVAL_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
        for line in p.stdout:
            lf.write(line)
            if any(k in line for k in ("Evaluating model", "Test:", "Time taken", "Error", "Traceback", "out of memory")):
                print("   ", line.rstrip()[:300], flush=True)
            elif time.time() - last > 600:
                print(f"    ... {dataset} 生成中 {(time.time() - t0) / 60:.0f} 分", flush=True)
                last = time.time()
        rc = p.wait()
    if rc != 0:
        print(open(log_path).read()[-3000:])
        raise RuntimeError(f"math_eval_budget.py が失敗 (rc={rc})。ログ: {log_path}")
    res = read_official_math_result(out_dir, dataset)
    assert res, f"結果ファイルが見つからない: {out_dir}"
    res["elapsed_sec"] = round(time.time() - t0)
    return res


def cot_public_step_dir(step):
    """公開 CoT 学生の stepN/（HF 形式）を取得してローカルパスを返す（vLLM は HF の subfolder を直接読めない）"""
    d = f"{MODELS_DIR}/cot_{MODEL_KEY}_{MODEL_INFO['cot_revision'][:8]}"
    snapshot_download(repo_id=MODEL_INFO["cot_repo"], revision=MODEL_INFO["cot_revision"], allow_patterns=[f"step{step}/*"], local_dir=d)
    return f"{d}/step{step}"


def paper_eval_targets():
    targets = []
    if PAPER_EVAL_BASE:
        targets.append(dict(kind="base", name=f"{MODEL_KEY}-Base", step=0))
    for s in PAPER_EVAL_COT_STEPS:
        targets.append(dict(kind="cot", name=f"{MODEL_KEY}_Math-CoT(public)", step=int(s)))
    runs = PAPER_EVAL_AO_RUNS or [rid for rid, r in globals().get("TRAIN_RESULTS", {}).items() if r["exit_code"] == 0]
    for rid in runs:
        done = [c["step"] for c in list_hf_checkpoints(rid) if c["complete"]]   # 重みだけの step も評価できる
        steps = [s for s in PAPER_EVAL_STEPS if s in done] if PAPER_EVAL_AO_STEPS == "paper" else [int(s) for s in PAPER_EVAL_AO_STEPS]
        missing = [s for s in steps if s not in done]
        assert not missing, f"{rid}: HF に無い step {missing}"
        targets += [dict(kind="ao", name=rid, run_id=rid, step=s) for s in steps]
    return targets


def paper_reference(kind, step, dataset):
    """論文の値（App. D）: Base なら Base の値、それ以外は同じ step の公開 CoT 学生の値。無ければ None"""
    ref, j = PAPER_REFERENCE.get(MODEL_KEY, {}), {"MATH500": 0, "AIME24": 1}.get(dataset)
    if j is None:
        return None
    if kind == "base":
        return ref.get("base", {}).get(dataset)
    row = ref.get("Math-CoT", {}).get(step)
    return None if row is None else row[j]


def evaluate_paper_target(t):
    """1 つのモデル（run と step）を論文と同じ方法で評価する。AO は HF の checkpoint を変換してから評価し、変換物は評価後に消せる"""
    tag = f"{t['name']}_step{t['step']}"
    created = []
    if t["kind"] == "base":
        model_dir = get_base_model_local()
    elif t["kind"] == "cot":
        model_dir = cot_public_step_dir(t["step"])
    else:
        _ck = f"{CKPT_DIR}/{t['run_id']}/global_step_{t['step']}"
        _had_ckpt = os.path.isdir(_ck)
        model_dir = merge_checkpoint(t["run_id"], t["step"])
        created = [model_dir] + ([] if _had_ckpt else [_ck])
    row = dict(t, model_dir=model_dir, results={}, deviations=PAPER_EVAL_DEVIATIONS, evaluated_at=now_iso())
    for ds in PAPER_EVAL_DATASETS:
        print(f"  [{tag}] {ds} ...", flush=True)
        row["results"][ds] = run_official_math_eval(model_dir, ds, f"{EVAL_DIR}/paper/{tag}", f"{LOG_DIR}/eval/{tag}_{ds}.log")
        row["results"][ds]["paper_reference"] = paper_reference(t["kind"], t["step"], ds)   # Base は Base の論文値、他は同じ step の CoT 学生の論文値
    if PAPER_EVAL_DELETE_MERGED and t["kind"] == "ao":
        for d in created:
            shutil.rmtree(d, ignore_errors=True)
    return row


PAPER_EVAL_RESULTS = globals().get("PAPER_EVAL_RESULTS", [])
print("=== 9-a 論文と同じ評価 ===")
print(f"  スクリプト: evaluation/math_eval/math_eval_budget.py（無変更）| データ: {PAPER_EVAL_DATASETS} | "
      f"MATH500 avg@3, AIME24 avg@10, temperature 0.6, top_p 0.95, 最大 32768 token, math-verify")
print("  論文との差:", PAPER_EVAL_DEVIATIONS)
if not PAPER_EVAL_ENABLED:
    print("  PAPER_EVAL_ENABLED=False のため実行しない（学習が終わってから True にする）")
else:
    assert N_GPUS >= 1, "GPU が必要"
    _eval_env = ensure_eval_env()
    ensure_official_eval_paths()
    _targets = paper_eval_targets()
    print(f"  評価対象 {len(_targets)} 件: {[(t['name'], t['step']) for t in _targets]}")
    for _t in _targets:
        _row = evaluate_paper_target(_t)
        _row["eval_env"] = _eval_env.get("versions")
        PAPER_EVAL_RESULTS.append(_row)
        with open(f"{RECORD_DIR}/paper_eval_results.json", "w") as f:
            json.dump(PAPER_EVAL_RESULTS, f, indent=2, ensure_ascii=False)
    print(f"\n  {'model':60s} {'step':>5s} " + " ".join(f"{d + ' avg@' + str(PAPER_EVAL_K[d]):>14s} {'(論文 CoT)':>10s} {'長さ':>7s}" for d in PAPER_EVAL_DATASETS))
    for _r in PAPER_EVAL_RESULTS:
        _cells = []
        for d in PAPER_EVAL_DATASETS:
            _x = _r["results"][d]
            _ref = _x.get("paper_reference")
            _cells.append(f"{_x[f'avg@{PAPER_EVAL_K[d]}']:14.1f} {('' if _ref is None else f'{_ref:.1f}'):>10s} {_x['avg_length_tokens']:7.0f}")
        print(f"  {_r['name'][:60]:60s} {_r['step']:5d} " + " ".join(_cells))
    print("  （論文 CoT）は同じ step の公開 CoT 学生の論文値（App. D。Base の行は Base の論文値）。NoCoT の値は PAPER_REFERENCE を参照")
    if HF_API is not None and HF_CKPT_REPO_ID:
        _ts = time.strftime("%Y%m%d-%H%M%S")
        for _rid in sorted({r["run_id"] for r in PAPER_EVAL_RESULTS if r["kind"] == "ao"}):
            _p = f"{RECORD_DIR}/paper_eval_{_rid}_{_ts}.json"
            json.dump([r for r in PAPER_EVAL_RESULTS if r.get("run_id") == _rid], open(_p, "w"), indent=2, ensure_ascii=False)
            HF_API.upload_file(path_or_fileobj=_p, path_in_repo=f"runs/{_rid}/paper_eval_{_ts}.json", repo_id=HF_CKPT_REPO_ID, repo_type="model",
                               commit_message=f"{_rid} paper-protocol evaluation {_ts}")   # 時刻付きの新しいファイル（上書きしない）
        print("  評価結果を HF の runs/<run_id>/paper_eval_<時刻>.json に保存した")


# ---- 9-b 任意の dev 評価（論文には無い。探索候補を選ぶ場合だけ使う）--------------------------------------------
# 生成と採点は 9-a と同じ: 公式スクリプトと同じ prompt（"\n Please reason step by step, ..."）、vLLM、temperature 0.6, top_p 0.95,
# 最大 32768 token, seed 1234, stop = eos、math-verify（gold を \boxed{} で包み、応答全体を parse）
DEV_RUNNER_SRC = r'''
import json, sys


def score(gold, responses):
    """math_eval_budget.py の process_single_result_no_truncation と同じ採点"""
    from math_verify import parse, verify
    gold = str(gold)
    gold_parsed = parse(gold) if "boxed" in gold else parse("\\boxed{{{}}}".format(gold))
    flags = []
    for r in responses:
        try:
            p = parse(r)
        except Exception:
            p = r
        flags.append(bool(verify(gold_parsed, p)))
    return flags


def main(cfg_path):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    cfg = json.load(open(cfg_path))
    rows = [json.loads(l) for l in open(cfg["data"]) if l.strip()]
    tok = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True, trust_remote_code=True)
    prompts = [tok.apply_chat_template([{"role": "user", "content": "{}\n Please reason step by step, and put your final answer within \\boxed{{}}.".format(r["question"])}],
                                       add_generation_prompt=True, tokenize=False) for r in rows]
    llm = LLM(cfg["model"], gpu_memory_utilization=0.85, tensor_parallel_size=1, trust_remote_code=True)
    sp = SamplingParams(temperature=0.6, max_tokens=32768, n=cfg["n"], top_p=0.95, seed=1234)
    sp.stop_token_ids = [tok.eos_token_id]
    outs = llm.generate(prompts, sampling_params=sp, use_tqdm=True)
    per = []
    for r, o in zip(rows, outs):
        resp = [x.text for x in o.outputs]
        flags = score(r["answer"], resp)
        per.append(dict(question=r["question"], gold=r["answer"], responses=resp, correct=flags, tokens=[len(x.token_ids) for x in o.outputs]))
    avg = sum(sum(p["correct"]) / len(p["correct"]) for p in per) / max(1, len(per))
    json.dump(dict(avg_at_n=100 * avg, n=cfg["n"], rows=len(per), per_sample=per), open(cfg["out"], "w"), ensure_ascii=False)
    print("DEV_RESULT", json.dumps(dict(avg_at_n=100 * avg, n=cfg["n"], rows=len(per))))


if __name__ == "__main__":
    main(sys.argv[1])
'''


def load_dev_rows():
    if not DEV_EVAL_SOURCE:
        return None
    if os.path.isfile(DEV_EVAL_SOURCE):
        rows = [json.loads(l) for l in open(DEV_EVAL_SOURCE) if l.strip()]
    else:
        from datasets import load_dataset
        rows = [dict(r) for r in load_dataset(DEV_EVAL_SOURCE, split=DEV_EVAL_SPLIT, revision=DEV_EVAL_REVISION)]
    out = [dict(question=r[DEV_EVAL_QUESTION_KEY], answer=(r[DEV_EVAL_ANSWER_KEY][0] if isinstance(r[DEV_EVAL_ANSWER_KEY], list) else r[DEV_EVAL_ANSWER_KEY])) for r in rows]
    return out[:DEV_EVAL_MAX_ROWS] if DEV_EVAL_MAX_ROWS else out


def evaluate_dev_target(target, rows):
    tag = target["name"].replace("/", "_")
    d = f"{EVAL_DIR}/dev/{tag}"
    os.makedirs(d, exist_ok=True)
    data = f"{d}/dev.jsonl"
    with open(data, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    runner = f"{EVAL_DIR}/dev_runner.py"
    open(runner, "w").write(DEV_RUNNER_SRC)
    cfg = dict(model=target["path"], data=data, n=DEV_EVAL_N, out=f"{d}/result.json")
    json.dump(cfg, open(f"{d}/cfg.json", "w"))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    env.update(VLLM_ATTENTION_BACKEND="XFORMERS", VLLM_WORKER_MULTIPROC_METHOD="spawn", CUDA_VISIBLE_DEVICES="0")
    t0 = time.time()
    r = subprocess.run([EVAL_PY, runner, f"{d}/cfg.json"], capture_output=True, text=True, env=env)
    open(f"{d}/log.txt", "w").write(r.stdout + r.stderr)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise RuntimeError(f"dev 評価に失敗: {target['name']}")
    res = json.loads([l for l in r.stdout.splitlines() if l.startswith("DEV_RESULT ")][-1][len("DEV_RESULT "):])
    res.update(name=target["name"], path=target["path"], dev_source=DEV_EVAL_SOURCE, dev_split=DEV_EVAL_SPLIT, dev_revision=DEV_EVAL_REVISION,
               elapsed_sec=round(time.time() - t0), scorer="9-a と同じ生成条件・math-verify 採点（avg@n）")
    print(f"  {target['name']}: avg@{res['n']} = {res['avg_at_n']:.2f} (rows {res['rows']}, {res['elapsed_sec']}s)")
    return res


DEV_EVAL_RESULTS = globals().get("DEV_EVAL_RESULTS", [])
print("\n=== 9-b dev 評価（論文には無い追加手順）===")
if not DEV_EVAL_ENABLED:
    print("  DEV_EVAL_ENABLED=False。論文は dev を使わず、各条件を test ベンチマークの step ごとの推移で報告している（9-a と同じ）")
else:
    assert N_GPUS >= 1
    assert DEV_EVAL_IS_INDEPENDENT, "DEV_EVAL_IS_INDEPENDENT=True を設定する前に、dev が学習データ・最終 test と独立であることを確認する"
    _forbidden = {DATASET_REPO, DATASET_ORIGIN_POOL, os.path.abspath(AO_PARQUET), os.path.abspath(RAW_PARQUET)}
    assert DEV_EVAL_SOURCE not in _forbidden and (not os.path.isfile(DEV_EVAL_SOURCE) or os.path.abspath(DEV_EVAL_SOURCE) not in _forbidden), \
        "学習データ (Math-CoT-20k / OpenR1-Math-220k / AO parquet) を dev にしない"
    ensure_eval_env()
    DEV_ROWS = load_dev_rows()
    print(f"  dev: {DEV_EVAL_SOURCE} split={DEV_EVAL_SPLIT} rows={len(DEV_ROWS)}")
    _targets = list(DEV_EVAL_TARGETS)
    if not _targets:
        for _rid, _res in globals().get("TRAIN_RESULTS", {}).items():
            if _res["exit_code"] == 0:
                _st = _res["spec"]["total_steps"]
                _targets.append(dict(name=f"{_rid}@step{_st}", path=merge_checkpoint(_rid, _st)))
    for _t in _targets:
        DEV_EVAL_RESULTS.append(evaluate_dev_target(_t, DEV_ROWS))
    with open(f"{RECORD_DIR}/dev_eval_results.json", "w") as f:
        json.dump(DEV_EVAL_RESULTS, f, indent=2, ensure_ascii=False)
    print("  dev 集計（候補全体を評価し終えるまで『確定』としない）:")
    for _r in sorted(DEV_EVAL_RESULTS, key=lambda x: -x["avg_at_n"]):
        print(f"    {_r['avg_at_n']:.2f}  {_r['name']}")
