# @title 9. 論文と同じ評価（公式 evaluation/math_eval/math_eval_budget.py を無変更で実行。各 step の推移を記録し、選択はしない）
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
PAPER_EVAL_DEVIATIONS = ["tensor_parallel_size 1（論文と公式スクリプトは 2 GPU で 2）と GPU の種類（論文は H200）。数値がわずかに変わると temperature 0.6 の"
                         "サンプリング結果も変わるので、論文値とはサンプリングの揺らぎの範囲（AIME24 は 30 問なので数ポイント）で異なり得る。"
                         "CoT 学生と厳密に比べるときは同じ環境で CoT も評価する（PAPER_EVAL_COT_STEPS）"]
# ipykernel は MPLBACKEND を inline 用の値に設定する。評価用の環境には matplotlib_inline が無く、公式スクリプトの import matplotlib が失敗するので
# subprocess には描画しない既定の Agg を渡す（評価の値には影響しない）
EVAL_SUBPROCESS_ENV = {"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}


def eval_subprocess_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV") and not k.startswith("UV_")}
    env.update(EVAL_SUBPROCESS_ENV, **extra)
    return env


def ensure_eval_env():
    """論文の数学評価用の環境（Python 3.12 + vLLM 0.8.5 + 公式 pin）を作る。学習用の公式環境とは別"""
    if GPU_PROFILE != "official":
        raise SystemExit(f"GPU プロファイル {GPU_PROFILE}: vLLM 0.8.5（公式 pin）はこの GPU 用のコードを持たない（sm_70〜sm_90 のみ）。"
                         "評価は A100 のランタイムで行う（学習済みの checkpoint は HF にある）")
    pins = ENV_BOOTSTRAP["eval_pip_pinned"]
    cons = ENV_BOOTSTRAP["eval_constraints"]   # 公式 requirements.txt（矛盾する protobuf / typer を除く）
    h = hashlib.sha256((json.dumps(pins) + open(cons).read()).encode()).hexdigest()[:16]
    marker = f"{EVAL_ENV_DIR}/ao_eval_env_ok.json"
    if os.path.isfile(marker) and json.load(open(marker)).get("hash") == h and os.path.isfile(EVAL_PY):
        return json.load(open(marker))
    uv_env = eval_subprocess_env(UV_CACHE_DIR=ENV_BOOTSTRAP["uv_cache_dir"], UV_PYTHON_INSTALL_DIR=ENV_BOOTSTRAP["uv_python_install_dir"])
    uv = ENV_BOOTSTRAP["uv_cmd"]
    t0 = time.time()
    shutil.rmtree(EVAL_ENV_DIR, ignore_errors=True)
    for cmd in (uv + ["venv", "--quiet", "--python", "3.12", "--python-preference", "only-managed", EVAL_ENV_DIR],
                uv + ["pip", "install", "--quiet", "--python", EVAL_PY, "-c", cons] + pins):
        r = subprocess.run(cmd, capture_output=True, text=True, env=uv_env)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:])
            raise RuntimeError("評価用の環境の作成に失敗")
    r = subprocess.run([EVAL_PY, "-c", "import json, vllm, torch, transformers, math_verify, xformers, matplotlib, pandas, datasets; "
                                       "print(json.dumps(dict(vllm=vllm.__version__, torch=torch.__version__, transformers=transformers.__version__, "
                                       "xformers=xformers.__version__, datasets=datasets.__version__)))"],
                       capture_output=True, text=True, env=uv_env)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise RuntimeError("評価用の環境の import 確認に失敗")
    freeze = subprocess.run(uv + ["pip", "freeze", "--python", EVAL_PY], capture_output=True, text=True, env=uv_env).stdout
    with open(f"{RECORD_DIR}/eval_env_freeze.txt", "w") as f:   # 評価用環境の全パッケージの版（記録）
        f.write(freeze)
    info = dict(hash=h, pins=pins, versions=json.loads(r.stdout.strip().splitlines()[-1]), created_sec=round(time.time() - t0),
                freeze_file=f"{RECORD_DIR}/eval_env_freeze.txt")
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
    save_name, budget = PAPER_EVAL_OUTPUT_KEYS[dataset]
    for f in glob.glob(f"{out_dir}/new_outputs/*/{save_name}_budget{budget}/no_instruct_512.json"):
        try:
            json.load(open(f))
        except ValueError:   # 生成結果の書き込み中に止まった壊れたファイル。残すと公式スクリプトが毎回読み込みで失敗する
            print(f"    壊れた生成結果を削除して生成し直す: {f}")
            os.remove(f)
    check_gpu_free_for_vllm()
    cmd = [EVAL_PY, "math_eval_budget.py", f"--model_path={model_dir}", f"--dataset={PAPER_EVAL_DATASET_PATHS[dataset]}", "--types=no_instruct",
           f"--output_dir={out_dir}", "--num_gpus=1"]
    env = eval_subprocess_env(VLLM_ATTENTION_BACKEND="XFORMERS", VLLM_WORKER_MULTIPROC_METHOD="spawn", CUDA_VISIBLE_DEVICES="0",   # run_math.sh と同じ
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


def check_gpu_free_for_vllm(max_used_frac=0.10):
    """vLLM（V0）は GPU 全体の 85% を確保しようとし、他のプロセスが使っている分を差し引かない。他が 10% 以上使っていれば止める"""
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    if r.returncode != 0:
        return
    used, total = (float(x) for x in r.stdout.splitlines()[0].split(","))
    if used / total > max_used_frac:
        raise SystemExit(f"GPU メモリを他のプロセスが {used:.0f}/{total:.0f} MiB 使っている。学習や他の評価が終わってから実行する"
                         "（このカーネルでモデルを GPU に載せた場合はセル 1 から実行し直す）")


def cot_public_step_dir(step):
    """公開 CoT 学生の stepN/（HF 形式）を取得してローカルパスを返す（vLLM は HF の subfolder を直接読めない）"""
    d = f"{MODELS_DIR}/cot_{MODEL_KEY}_{MODEL_INFO['cot_revision'][:8]}"
    snapshot_download(repo_id=MODEL_INFO["cot_repo"], revision=MODEL_INFO["cot_revision"], allow_patterns=[f"step{step}/*"], local_dir=d)
    assert os.path.isfile(f"{d}/step{step}/config.json"), f"{MODEL_INFO['cot_repo']} に step{step} が無い（公開されている step: 10, 20, 40, 80, 160, 320, 480, 640）"
    return f"{d}/step{step}"


def paper_eval_targets():
    targets = []
    if PAPER_EVAL_BASE:
        targets.append(dict(kind="base", name=f"{MODEL_KEY}-Base", step=0))
    for s in PAPER_EVAL_COT_STEPS:
        targets.append(dict(kind="cot", name=f"{MODEL_KEY}_Math-CoT(public)", step=int(s)))
    runs = PAPER_EVAL_AO_RUNS or [rid for rid, r in globals().get("TRAIN_RESULTS", {}).items() if r["exit_code"] == 0]
    if not runs:
        print("  WARN: 評価する AO run が無い。このセッションで学習していない場合（切断後など）は PAPER_EVAL_AO_RUNS に run_id を指定する")
    for rid in runs:
        done = [c["step"] for c in list_hf_checkpoints(rid) if c["complete"]]   # 重みだけの step も評価できる
        steps = [s for s in PAPER_EVAL_STEPS if s in done] if PAPER_EVAL_AO_STEPS == "paper" else [int(s) for s in PAPER_EVAL_AO_STEPS]
        missing = [s for s in steps if s not in done]
        assert not missing, f"{rid}: HF に無い step {missing}"
        cond = get_manifest(rid, steps[0])["run_config"].get("paper_condition") if steps else None
        targets += [dict(kind="ao", name=rid, run_id=rid, step=s, paper_condition=cond) for s in steps]
    return targets


def paper_reference(kind, step, dataset, paper_condition=None):
    """論文の値（App. D）: Base なら Base の値、それ以外は同じ step の公開 CoT 学生の値。無ければ None。
    1.7B/4B の CoT 学生は論文の既定条件（lr 5e-5, 8 epoch, cosine）しか無いので、AO も既定条件の run だけに並べる"""
    ref, j = PAPER_REFERENCE.get(MODEL_KEY, {}), {"MATH500": 0, "AIME24": 1}.get(dataset)
    if j is None:
        return None
    if kind == "base":
        return ref.get("base", {}).get(dataset)
    if kind == "ao" and not str(paper_condition or "").startswith("default"):
        return None
    row = ref.get("Math-CoT", {}).get(step)
    return None if row is None else row[j]


def evaluate_paper_target(t):
    """1 つのモデル（run と step）を論文と同じ方法で評価する。AO は HF の checkpoint を変換してから評価し、この関数が作った変換物だけを評価後に消せる"""
    tag = f"{t['name']}_step{t['step']}"
    out_dir = f"{EVAL_DIR}/paper/{tag}"
    row = dict(t, results={}, deviations=PAPER_EVAL_DEVIATIONS, evaluated_at=now_iso())
    cached = {ds: read_official_math_result(out_dir, ds) for ds in PAPER_EVAL_DATASETS}
    created, model_dir = [], None
    if not all(cached.values()):
        if t["kind"] == "base":
            model_dir = get_base_model_local()
        elif t["kind"] == "cot":
            model_dir = cot_public_step_dir(t["step"])
        else:
            _ck = f"{CKPT_DIR}/{t['run_id']}/global_step_{t['step']}"
            _merged = f"{WORK_DIR}/merged/{t['run_id']}/merged_step{t['step']}"
            _had = (os.path.isdir(_ck), os.path.isdir(_merged))
            model_dir = merge_checkpoint(t["run_id"], t["step"])
            created = ([] if _had[1] else [model_dir]) + ([] if _had[0] else [_ck])
    row["model_dir"] = model_dir
    for ds in PAPER_EVAL_DATASETS:
        print(f"  [{tag}] {ds} ...", flush=True)
        row["results"][ds] = cached[ds] or run_official_math_eval(model_dir, ds, out_dir, f"{LOG_DIR}/eval/{tag}_{ds}.log")
        row["results"][ds]["paper_reference"] = paper_reference(t["kind"], t["step"], ds, t.get("paper_condition"))
    if PAPER_EVAL_DELETE_MERGED and t["kind"] == "ao":
        for d in created:
            shutil.rmtree(d, ignore_errors=True)
    return row


def save_paper_eval_row(row):
    """評価ごとに保存する（切断で失わないよう HF にも時刻付きの新しいファイルで置く。上書きしない）"""
    with open(f"{RECORD_DIR}/paper_eval_results.json", "w") as f:
        json.dump(PAPER_EVAL_RESULTS, f, indent=2, ensure_ascii=False)
    if HF_API is not None and HF_CKPT_REPO_ID:
        ts = time.strftime("%Y%m%d-%H%M%S")
        tag = f"{row['name']}_step{row['step']}".replace("/", "_")
        path_in_repo = f"runs/{row['run_id']}/paper_eval/step{row['step']}_{ts}.json" if row["kind"] == "ao" else f"paper_eval/{tag}_{ts}.json"
        local = f"{RECORD_DIR}/paper_eval_{tag}_{ts}.json"
        json.dump(row, open(local, "w"), indent=2, ensure_ascii=False)
        HF_API.upload_file(path_or_fileobj=local, path_in_repo=path_in_repo, repo_id=HF_CKPT_REPO_ID, repo_type="model",
                           commit_message=f"paper-protocol evaluation {tag}")


PAPER_EVAL_RESULTS = globals().get("PAPER_EVAL_RESULTS", [])
print("=== 論文と同じ評価 ===")
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
        _key = (_row["kind"], _row["name"], _row["step"])
        PAPER_EVAL_RESULTS[:] = [r for r in PAPER_EVAL_RESULTS if (r["kind"], r["name"], r["step"]) != _key] + [_row]   # 再実行で重複させない
        save_paper_eval_row(_row)
    print(f"\n  {'model':60s} {'step':>5s} " + " ".join(f"{d + ' avg@' + str(PAPER_EVAL_K[d]):>14s} {'(論文 CoT)':>10s} {'長さ':>7s}" for d in PAPER_EVAL_DATASETS))
    for _r in PAPER_EVAL_RESULTS:
        _cells = []
        for d in PAPER_EVAL_DATASETS:
            _x = _r["results"].get(d)
            if _x is None:
                _cells.append(f"{'-':>14s} {'':>10s} {'':>7s}")
                continue
            _ref = _x.get("paper_reference")
            _cells.append(f"{_x[f'avg@{PAPER_EVAL_K[d]}']:14.1f} {('' if _ref is None else f'{_ref:.1f}'):>10s} {_x['avg_length_tokens']:7.0f}")
        print(f"  {_r['name'][:60]:60s} {_r['step']:5d} " + " ".join(_cells))
    print("  （論文 CoT）は同じ step の公開 CoT 学生の論文値（App. D。Base の行は Base の論文値。論文の既定条件以外の AO run には並べない）。NoCoT は PAPER_REFERENCE")
    print("  結果は評価ごとに records/paper_eval_results.json と HF（runs/<run_id>/paper_eval/, paper_eval/）に保存した")
