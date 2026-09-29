# @title 4-a. 共通ヘルパー: 公式スクリプトの解析・run 仕様の組み立て（公式実装は変更せず既存引数だけを渡す）
import re, json, os, math, pathlib, random, hashlib, datetime
import pyarrow as pa, pyarrow.parquet as pq
from omegaconf import OmegaConf
from huggingface_hub import snapshot_download, HfApi

HF_API = HfApi() if HF_LOGGED_IN else None
if HF_CKPT_REPO_ID is None and HF_ACCOUNT_NAME:
    HF_CKPT_REPO_ID = f"{HF_ACCOUNT_NAME}/rethink-sft-ao-checkpoints"   # HF の whoami から決める（GitHub 名からは推測しない）
print("HF checkpoint repo:", HF_CKPT_REPO_ID, "(private=%s)" % HF_CKPT_PRIVATE, "| upload during training:", HF_UPLOAD_CHECKPOINTS)


def hf_repo_exists(repo_id, repo_type="model"):
    if HF_API is None or not repo_id:
        return False
    try:
        HF_API.repo_info(repo_id, repo_type=repo_type)
        return True
    except Exception:
        return False


def hf_run_paths(repo_id, run_id):
    """HF 上の runs/<run_id>/ 配下の global_step_* を列挙（完了マーカー ao_upload_verified.json の有無付き）"""
    if not hf_repo_exists(repo_id):
        return {}
    try:
        files = HF_API.list_repo_files(repo_id, repo_type="model")
    except Exception:
        return {}
    out = {}
    for f in files:
        m = re.match(rf"runs/{re.escape(run_id)}/global_step_(\d+)/(.*)$", f)
        if m:
            out.setdefault(int(m.group(1)), set()).add(m.group(2))
    return {s: dict(files=sorted(v), complete=("ao_upload_verified.json" in v)) for s, v in out.items()}


def parse_official_script(path):
    """training_scripts/*.sh から変数と torchrun の hydra 上書き引数を取り出す（$VAR は解決する）"""
    text = pathlib.Path(path).read_text()
    var = {}
    exported = []
    for m in re.finditer(r"^(export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$", text, re.M):
        is_export, k, v = bool(m.group(1)), m.group(2), m.group(3).strip()
        if v.startswith("$((") or v.startswith("$(") :
            continue
        v = v.strip().strip('"').strip("'")
        var[k] = v
        if is_export:
            exported.append(k)

    def resolve(v):
        for _ in range(3):
            v = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)",
                       lambda mm: var.get(mm.group(1) or mm.group(2), mm.group(0)), v)
        return v

    var = {k: resolve(v) for k, v in var.items()}
    m = re.search(r"torchrun(.*?)(?:>>|$)", text, re.S)
    assert m, "torchrun ブロックが見つからない"
    tokens = m.group(1).replace("\\\n", " ").split()
    overrides, torchrun_args, module = {}, [], None
    for i, t in enumerate(tokens):
        if t.startswith("--"):
            torchrun_args.append(resolve(t))
        elif t == "-m":
            module = tokens[i + 1]
        elif "=" in t and not t.startswith("-") and t != module:
            k, v = t.split("=", 1)
            overrides[k] = resolve(v.strip('"').strip("'"))
    env = {k: var[k] for k in exported if k not in ("WANDB_API_KEY", "PYTHONPATH")}
    return dict(path=str(path), vars=var, overrides=overrides, torchrun_args=torchrun_args, module=module, env=env)


def hydra_value(v):
    """hydra の CLI 値（文字列）を Python 値へ。OmegaConf で dataset config を組むときに使う"""
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    s = str(v).strip()
    if s in ("null", "None"):
        return None
    if s in ("True", "true"):
        return True
    if s in ("False", "false"):
        return False
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if re.fullmatch(r"-?\d*\.\d+(e-?\d+)?|-?\d+e-?\d+", s):
        return float(s)
    if s.startswith("[") and s.endswith("]"):
        return json.loads(s.replace("'", '"'))
    return s


def fmt_lr(lr):
    mant, exp = f"{float(lr):e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    return f"{mant}e{int(exp)}"


OFFICIAL_SCRIPT_PATH = f"{REPO_DIR}/training_scripts/{MODEL_INFO['official_script']}"
OFFICIAL = parse_official_script(OFFICIAL_SCRIPT_PATH)
YAML_DEFAULTS = OmegaConf.load(f"{REPO_DIR}/verl/trainer/config/sft_trainer.yaml")
OFFICIAL_MAX_LENGTH = hydra_value(OFFICIAL["overrides"]["data.max_length"])
OFFICIAL_TBS = hydra_value(OFFICIAL["overrides"]["data.train_batch_size"])
OFFICIAL_MICRO_BSZ = hydra_value(OFFICIAL["overrides"]["data.micro_batch_size_per_gpu"])
OFFICIAL_LR = float(OFFICIAL["overrides"]["optim.lr"])
OFFICIAL_EPOCHS = hydra_value(OFFICIAL["overrides"]["trainer.total_epochs"])

print("公式スクリプト:", OFFICIAL_SCRIPT_PATH)
print("  module:", OFFICIAL["module"], "| torchrun args:", OFFICIAL["torchrun_args"])
print("  env exports:", OFFICIAL["env"])
print("  hydra overrides:")
for _k, _v in OFFICIAL["overrides"].items():
    print(f"    {_k} = {_v}")
print("  yaml 既定（スクリプトで上書きされない主要値）: weight_decay=%s warmup_steps_ratio=%s clip_grad=%s seed=%s strategy=%s grad_ckpt=%s lora_rank=%s"
      % (YAML_DEFAULTS.optim.weight_decay, YAML_DEFAULTS.optim.warmup_steps_ratio, YAML_DEFAULTS.optim.clip_grad, YAML_DEFAULTS.trainer.seed,
         YAML_DEFAULTS.model.strategy, YAML_DEFAULTS.model.enable_gradient_checkpointing, YAML_DEFAULTS.model.lora_rank))
assert hydra_value(YAML_DEFAULTS.model.lora_rank) == 0 and "lora_rank" not in OFFICIAL["overrides"], "Full-parameter SFT のはずが LoRA 設定がある"
assert OFFICIAL_LR == OFFICIAL_BASELINE["lr"] and OFFICIAL_EPOCHS == OFFICIAL_BASELINE["epochs"], "設定セルの OFFICIAL_BASELINE と公式スクリプトが一致しない"


def get_base_model_local():
    """Base モデルを pin した revision で取得し、ローカルパスを返す（model.partial_pretrain に渡す = revision 固定）"""
    d = f"{MODELS_DIR}/{MODEL_KEY}-Base_{MODEL_INFO['base_revision'][:8]}"
    p = snapshot_download(repo_id=MODEL_INFO["base_repo"], revision=MODEL_INFO["base_revision"], local_dir=d)
    return p.rstrip("/")


def steps_per_epoch(n_rows, tbs, n_gpus):
    """DistributedSampler(drop_last) + DataLoader(drop_last) と同じ計算"""
    per_rank_rows = n_rows // n_gpus
    per_rank_bs = tbs // n_gpus
    return per_rank_rows // per_rank_bs


# ---- 公式 8 GPU の micro batch 構成と勾配の大きさを N GPU で再現する -----------------------------
TRAIN_VIEW_VERSION = 1

# 再現しても残る差（公式コードを変えずには揃えられない）。実験記録と Model Card に書く
def residual_differences(n_gpus, max_length, emulated):
    out = []
    if n_gpus != OFFICIAL_WORLD_SIZE and emulated:
        out.append("勾配の加算の順序と精度: 公式は各 GPU で 8 micro batch 分を bf16 で加算し、GPU 間の平均は fp32。"
                   f"{n_gpus} GPU では {64 // n_gpus} micro batch 分を bf16 で加算する。丸め誤差の大きさが変わる"
                   "（CPU 上の公式 trainer と小型モデルでの検証: 6 step 後のパラメータ差は公式の移動量に対して 1 GPU で 1.5〜3.6%、"
                   "2 GPU で 0.7〜1.3%。並べ替えと loss 倍率なしの 1 GPU は 7〜14%。fp32 では 1e-6）")
        out.append(f"ログの train/loss: adv-only では token 平均が fp32 で計算され、vanilla では bf16。{OFFICIAL_WORLD_SIZE}/{n_gpus} 倍に戻した値は"
                   "公式ログと micro batch ごとの bf16 丸め 1 回分（最大 0.4% 程度）ずれ得る。勾配は影響を受けない")
    out.append("GPU の種類: 公式は H200。Colab の GPU では選ばれる CUDA カーネルが変わり、丸め誤差が変わる")
    if GPU_PROFILE_DEVIATION:
        out.append(f"torch の版（GPU プロファイル {GPU_PROFILE}）: {GPU_PROFILE_DEVIATION}")
    out.append("flash-attn の backward は非決定的（公式の学習も同じ条件）")
    if int(max_length) != int(OFFICIAL_MAX_LENGTH):
        out.append(f"data.max_length {max_length}（公式 {OFFICIAL_MAX_LENGTH}）: padding の量だけが変わる。padding は loss_mask と attention mask で除外されるので"
                   "loss と勾配は数学的に同じ（行列積の形が変わるので丸め誤差は変わる）")
    if n_gpus != OFFICIAL_WORLD_SIZE:
        out.append(f"train/grad_norm のログ: 公式 trainer は各 GPU の shard ノルムの平均を記録する（全体ノルムではない。CPU 上の公式 trainer で確認）。"
                   f"公式は {OFFICIAL_WORLD_SIZE} 分割、この run は {n_gpus} 分割の平均なので、公式ログとは直接比べられない（clip は正しい全体ノルムで行われる）")
    return out


def emulate_official_order(n_rows, official_world=8, actual_world=1, global_batch=256, micro=4):
    """公式 trainer（DistributedSampler(shuffle=False, drop_last=True) + StatefulDataLoader(batch=global_batch/N) + split(micro)）を
    N GPU で動かしたとき、各 step の各 micro batch が公式 official_world GPU の run と同じ行の組になる並び順を返す。
    戻り値 order: 学習用 file の p 行目 = 元 file の order[p] 行目。
      公式:  step k, rank r, micro t, 要素 i -> 元の行 B*k + r + W*(M*t + i)
      N GPU: step k, rank q, 位置 j        -> file 位置 B*k + N*j + q（j = (r_local*T + t)*M + i, r = q*G + r_local）
    256 行の block 内だけで並べ替えるので、先頭 m*256 行の試走 subset でも全体の先頭 m step と同じ micro batch になる。
    末尾の端数行（n_rows % 256）はどちらの run でも使われないので動かさない。"""
    W, N, B, M = int(official_world), int(actual_world), int(global_batch), int(micro)
    assert W >= 1 and N >= 1 and W % N == 0, f"actual_world {N} must divide official_world {W}"
    assert B % W == 0 and (B // W) % M == 0, f"global_batch {B} / official_world {W} / micro {M} inconsistent"
    G = W // N              # 1 つの実 GPU が受け持つ公式 rank の数
    T = B // W // M         # 公式 1 rank あたりの micro batch 数（256/8/4 = 8）
    S = n_rows // B         # 1 epoch の step 数（W でも N でも同じ）
    order = list(range(n_rows))
    for k in range(S):
        base = B * k
        for q in range(N):
            for r_local in range(G):
                r = q * G + r_local
                for t in range(T):
                    for i in range(M):
                        j = (r_local * T + t) * M + i
                        order[base + N * j + q] = base + r + W * (M * t + i)
    return order


def prepare_train_file(source_path, source_sha256, n_gpus):
    """学習に渡す parquet を返す: (path, sha256, view_meta)。
    N=8 または EMULATE_OFFICIAL_WORLD_SIZE=False なら元の file をそのまま使う。
    それ以外は元の file（AO parquet または試走 subset）を読み、emulate_official_order で行を並べ替え、advantage 列を N/8 にした
    派生 file を別名で書く。元の file は変更しない。派生 file の各行は元の行と teacher_answer / message が同一で、ao_source_row で対応が分かる。"""
    off = dict(enabled=False, version=TRAIN_VIEW_VERSION, official_world=OFFICIAL_WORLD_SIZE, actual_world=int(n_gpus),
               importance_sampling_mode=OFFICIAL["overrides"].get("trainer.importance_sampling_mode", "vanilla"), advantage=1.0,
               logged_loss_scale=1.0, source_path=source_path, source_sha256=source_sha256, train_file=source_path, train_file_sha256=source_sha256)
    if int(n_gpus) == OFFICIAL_WORLD_SIZE or not EMULATE_OFFICIAL_WORLD_SIZE:
        return source_path, source_sha256, off
    W, N = OFFICIAL_WORLD_SIZE, int(n_gpus)
    assert W % N == 0, f"GPU 台数 {N} は {W} の約数でないため公式 {W} GPU の micro batch 構成を再現できない（1, 2, 4, 8 台で実行する）"
    micro = OFFICIAL_MICRO_BSZ
    assert off["importance_sampling_mode"] == "vanilla", "公式スクリプトが vanilla 以外のときはこの再現方法を使えない"
    tbl = pq.read_table(source_path)
    n = tbl.num_rows
    adv_col = tbl.column("advantage")
    assert all(v == 1.0 for v in adv_col.to_pylist()), "元の advantage 列が 1.0 以外を含む。loss の倍率で勾配を合わせる前提が崩れる"
    order = emulate_official_order(n, W, N, OFFICIAL_TBS, micro)
    adv = N / W                                     # 1/8, 1/4, 1/2: 2 のべき乗なので浮動小数点で誤差なく掛けられる
    view = tbl.take(pa.array(order, type=pa.int64()))
    view = view.set_column(view.schema.get_field_index("advantage"), view.schema.field("advantage"), pa.array([adv] * n, type=adv_col.type))
    stem = os.path.splitext(os.path.basename(source_path))[0]
    out_dir = os.path.join(os.path.dirname(source_path), "train_view")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{stem}.emul-w{W}-n{N}.parquet")
    pq.write_table(view, out)
    # 書いた file を読み戻して、行の対応と advantage を確認する
    rt = pq.read_table(out)
    assert "ao_source_row" in tbl.column_names, "元の file に ao_source_row 列が無い（セクション 3 で作った AO parquet かその先頭 subset を渡す）"
    src_ids = tbl.column("ao_source_row").to_pylist()
    assert rt.num_rows == n and rt.column("ao_source_row").to_pylist() == [src_ids[o] for o in order], "派生 file の行の対応が崩れている"
    for c in ("message", "teacher_answer", "answer"):
        if c in tbl.column_names:
            assert rt.column(c).equals(tbl.column(c).take(pa.array(order, type=pa.int64()))), f"派生 file の {c} 列が元の行と一致しない"
    assert set(rt.column("advantage").to_pylist()) == {adv}
    meta = dict(off, enabled=True, importance_sampling_mode="adv-only", advantage=adv, logged_loss_scale=adv, global_batch=OFFICIAL_TBS, micro=micro,
                n_rows=n, order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(), train_file=out, train_file_sha256=sha256_of(out))
    return out, meta["train_file_sha256"], meta


def official_micro_batches(file_source_rows, world, global_batch, micro):
    """公式 trainer と同じ DistributedSampler で world 台に配ったときの、各 step の micro batch（元の行番号の組）を返す。
    戻り値: steps[k] = sorted list of tuples（その step の全 rank の micro batch）"""
    import torch.utils.data as tud
    n = len(file_source_rows)
    per_rank_bs = global_batch // world
    steps = None
    for r in range(world):
        idx = list(iter(tud.DistributedSampler(range(n), num_replicas=world, rank=r, shuffle=False, drop_last=True)))
        n_steps = len(idx) // per_rank_bs
        if steps is None:
            steps = [[] for _ in range(n_steps)]
        for k in range(n_steps):
            b = idx[k * per_rank_bs:(k + 1) * per_rank_bs]
            for m in range(0, per_rank_bs, micro):
                steps[k].append(tuple(file_source_rows[i] for i in b[m:m + micro]))
    return [sorted(s) for s in steps]


def normalize_run(entry):
    """SEARCH_RUN_LIST の要素 (lr, epochs) / (lr, epochs, scheduler) を (lr, epochs, scheduler) にそろえる"""
    assert isinstance(entry, (list, tuple)) and len(entry) in (2, 3), \
        f"探索候補は (lr, epochs) か (lr, epochs, scheduler) のタプルで書く（旧形式の {{'lr': [...], 'epochs': [...]}} は使えない）: {entry!r}"
    lr, ep, *rest = entry
    sched = rest[0] if rest else "cosine"
    assert sched in ("cosine", "constant"), f"lr_scheduler は論文で使われた cosine / constant のみ: {sched}"
    return float(lr), int(ep), sched


def paper_condition_of(lr, epochs, scheduler):
    """論文の最適化条件（PAPER_OPTIMIZATION_CONDITIONS）に一致すれば、その出典を返す"""
    for c in PAPER_OPTIMIZATION_CONDITIONS:
        if abs(c["lr"] - float(lr)) < 1e-12 and c["epochs"] == int(epochs) and c["scheduler"] == scheduler:
            return c["paper"]
    return None


def make_run_id(kind, lr, epochs, tbs=None, suffix="", scheduler="cosine"):
    tbs = tbs or OFFICIAL_TBS
    base = f"{MODEL_KEY}_Math-AO-20k_lr{fmt_lr(lr)}_ep{epochs}_bs{tbs}" + ("_ConstLR" if scheduler == "constant" else "")   # 公式スクリプト名と同じ付け方
    return f"{base}_{kind}{suffix}"


def build_run_spec(kind, lr, epochs, data_path, data_sha256, n_rows, max_length, save_freq, run_id=None, n_gpus=None,
                   resume_from_path=None, total_training_steps=None, extra_note=None, upload_policy=None, lr_scheduler="cosine"):
    """公式スクリプトの上書き引数を起点に、AO 学習に必要な最小限の引数だけ差し替える。差分は changes に記録する"""
    n_gpus = n_gpus or N_GPUS
    assert n_gpus >= 1, "GPU がない"
    ov = dict(OFFICIAL["overrides"])
    changes = []

    def set_(k, v, reason):
        old = ov.get(k, "<yaml default>")
        v = str(v)
        if old != v:
            changes.append(dict(key=k, official=old, new=v, reason=reason))
        ov[k] = v

    run_id = run_id or make_run_id(kind, lr, epochs, suffix=RUN_ID_SUFFIX, scheduler=lr_scheduler)
    _paper_cond = paper_condition_of(lr, epochs, lr_scheduler)
    base_local = get_base_model_local()
    set_("model.partial_pretrain", base_local, f"同じ Base を revision {MODEL_INFO['base_revision'][:8]} で固定したローカルパス")
    train_path, train_sha, view = prepare_train_file(data_path, data_sha256, n_gpus)
    set_("data.train_files", train_path, "AO データ（teacher_answer 列を追加した Math-CoT-20k）" +
         (f"を公式 {OFFICIAL_WORLD_SIZE} GPU の micro batch 構成になるよう並べ替えた学習用 file" if view["enabled"] else ""))
    set_("data.val_files", train_path, "公式同様 train と同じ値（参照版 trainer では validation はコメントアウト）")
    set_("data.response_key", "teacher_answer", "AO target 列")
    if view["enabled"]:
        set_("trainer.importance_sampling_mode", "adv-only",
             f"loss = -advantage × log_prob。advantage 列 = {view['advantage']}（= {n_gpus}/{OFFICIAL_WORLD_SIZE}）で vanilla の loss を {view['advantage']} 倍し、"
             f"{n_gpus} GPU の勾配を公式 {OFFICIAL_WORLD_SIZE} GPU と同じ大きさにする（勾配・clip・AdamW の入力が丸め誤差の範囲で公式と同じ値になる）")
        changes.append(dict(key="学習用 file の行順 / advantage 列", official="Math-CoT-20k の行順 / 1.0",
                            new=f"公式 {OFFICIAL_WORLD_SIZE} GPU の micro batch 構成を再現する並べ替え / {view['advantage']}",
                            reason=f"{n_gpus} GPU でも各 step の各 micro batch（4 行）を公式 {OFFICIAL_WORLD_SIZE} GPU と同じ行の組にする。"
                                   "各 step で使う 256 行の集合と epoch ごとの順序は公式と同じ。元の AO parquet は変更しない"))
    set_("data.max_length", max_length, "AO は短いので全行が収まる長さへ短縮（時間短縮。公式は 20000）" if max_length != OFFICIAL_MAX_LENGTH else "公式値")
    _why = f"探索対象（論文の条件: {_paper_cond}）" if _paper_cond else "探索対象（論文に無い条件）"
    set_("optim.lr", fmt_lr(lr), _why if float(lr) != OFFICIAL_LR else "公式値")
    set_("trainer.total_epochs", epochs, _why if int(epochs) != OFFICIAL_EPOCHS else "公式値")
    set_("optim.lr_scheduler", lr_scheduler, _why if lr_scheduler != OFFICIAL["overrides"].get("optim.lr_scheduler", "cosine") else "公式値")
    set_("trainer.save_freq", save_freq, "checkpoint 周期（正整数）。保存は学習結果に影響しない" if int(save_freq) != hydra_value(OFFICIAL["overrides"]["trainer.save_freq"]) else "公式値")
    set_("trainer.checkpoint.save_contents", '["model","optimizer","extra"]', "再開に必要な状態一式を保存（公式 CoT は model のみ）")
    set_("trainer.checkpoint.load_contents", '["model","optimizer","extra"]', "再開時に一式を読む")
    set_("trainer.default_local_dir", f"{CKPT_DIR}/{run_id}", "run 別ディレクトリ")
    set_("trainer.experiment_name", run_id, "run ID")
    set_("trainer.project_name", WANDB_PROJECT, "wandb project")
    set_("trainer.nnodes", 1, "Colab 単一ノード")
    if MICRO_BATCH_OVERRIDE is not None:
        assert not view["enabled"], "公式 8 GPU の再現中は micro batch を変えられない"
        set_("data.micro_batch_size_per_gpu", MICRO_BATCH_OVERRIDE, "メモリ都合の micro batch 変更（global batch 256 は維持）。micro batch 構成と token の重みが公式と変わる")
    if resume_from_path:
        set_("trainer.resume_mode", "resume_path", "HF から取得した checkpoint から再開")
        set_("trainer.resume_from_path", resume_from_path, "再開元")
    else:
        set_("trainer.resume_mode", "disable", "新規 run（同じ Base から独立に開始）")
    if total_training_steps is not None:
        set_("trainer.total_training_steps", total_training_steps, "試走専用の step 上限（scheduler 期間も変わる。本学習では使わない）")

    tbs = hydra_value(ov["data.train_batch_size"])
    micro = hydra_value(ov["data.micro_batch_size_per_gpu"])
    assert tbs % n_gpus == 0 and (tbs // n_gpus) % micro == 0, f"train_batch_size {tbs} / n_gpus {n_gpus} / micro {micro} の整合が取れない"
    spe = steps_per_epoch(n_rows, tbs, n_gpus)
    total_steps = int(math.ceil(spe * float(epochs))) if total_training_steps is None else int(total_training_steps)
    warmup = int(total_steps * float(YAML_DEFAULTS.optim.warmup_steps_ratio))
    env = dict(OFFICIAL["env"]) if KEEP_OFFICIAL_ENV_VARS else {k: v for k, v in OFFICIAL["env"].items() if k != "CUDA_LAUNCH_BLOCKING"}
    if not KEEP_OFFICIAL_ENV_VARS:
        changes.append(dict(key="env:CUDA_LAUNCH_BLOCKING", official="1", new="<unset>", reason="KEEP_OFFICIAL_ENV_VARS=False（速度優先。公式は 1）"))
    env["WANDB_MODE"] = WANDB_MODE if (WANDB_MODE == "offline" or os.environ.get("WANDB_API_KEY")) else "offline"
    env["WANDB_DIR"] = f"{WORK_DIR}/wandb"
    env["VERL_SFT_LOGGING_LEVEL"] = "INFO"     # 再開ログ（checkpoint 読み込み等）を表示する。既存の環境変数
    changes.append(dict(key="env:WANDB_DIR / VERL_SFT_LOGGING_LEVEL / PYTHONUNBUFFERED", official="<unset>", new=f"{env['WANDB_DIR']} / INFO / 1",
                        reason="wandb 出力先・再開ログ表示・ログ即時出力（学習内容に影響しない環境変数）"))
    if n_gpus == OFFICIAL_WORLD_SIZE:
        _nproc_reason = "公式と同じ台数"
    elif view["enabled"]:
        _nproc_reason = (f"実 GPU 台数。global batch {tbs} は 1 GPU あたり {(tbs // n_gpus) // micro} micro batch の勾配蓄積で実現し、"
                         f"行の並べ替えと loss の {view['advantage']} 倍で公式 {OFFICIAL_WORLD_SIZE} GPU の勾配を再現")
    else:
        _nproc_reason = (f"実 GPU 台数。EMULATE_OFFICIAL_WORLD_SIZE=False のため、micro batch 構成と勾配の大きさ（{OFFICIAL_WORLD_SIZE / n_gpus:g} 倍）"
                         "が公式と異なる。clip_grad=1.0 の効き方も変わる")
    changes.append(dict(key="torchrun --nproc_per_node", official=f"{OFFICIAL_WORLD_SIZE} (README: 8 x H200)", new=str(n_gpus), reason=_nproc_reason))
    _saves = sorted(set([s for s in range(int(save_freq), total_steps + 1, int(save_freq))] + [total_steps]))
    upload_policy = upload_policy or HF_UPLOAD_STEPS
    if upload_policy == "all":
        _uploads = list(_saves)
        _full = list(_saves)
    else:   # 論文が評価した step + 再開用の間隔 + 最終 step
        _uploads = sorted(s for s in _saves if s in PAPER_EVAL_STEPS or s % RESUME_CKPT_EVERY == 0 or s == total_steps)
        # 再開用（optimizer 状態込み）は RESUME_CKPT_EVERY の倍数と最終 step。それ以外（公開 CoT と同じ分析用 step）は model_only なら重みだけ送る
        _full = _uploads if HF_ANALYSIS_STEP_CONTENT == "full" else [s for s in _uploads if s % RESUME_CKPT_EVERY == 0 or s == total_steps]
    residual = residual_differences(n_gpus, max_length, view["enabled"])
    if n_gpus != OFFICIAL_WORLD_SIZE and not view["enabled"]:
        residual = [f"【再現無効】micro batch 構成と勾配の大きさ（{OFFICIAL_WORLD_SIZE / n_gpus:g} 倍）が公式 {OFFICIAL_WORLD_SIZE} GPU と異なる"] + residual
    spec = dict(
        run_id=run_id, kind=kind, lr=float(lr), lr_str=fmt_lr(lr), epochs=epochs, lr_scheduler=lr_scheduler, paper_condition=_paper_cond,
        model_key=MODEL_KEY, base_repo=MODEL_INFO["base_repo"],
        base_revision=MODEL_INFO["base_revision"], base_local=base_local, cot_repo=MODEL_INFO["cot_repo"], cot_revision=MODEL_INFO["cot_revision"],
        cot_subfolder=MODEL_INFO["cot_subfolder"], teacher=TEACHER, data_path=data_path, data_sha256=data_sha256, n_rows=n_rows,
        dataset_repo=DATASET_REPO, dataset_revision=DATASET_REVISION, target_style=AO_TARGET_STYLE, max_length=int(max_length),
        official_max_length=OFFICIAL_MAX_LENGTH, tbs=tbs, micro_bsz=micro, n_gpus=n_gpus, grad_accum_micro_batches=(tbs // n_gpus) // micro,
        steps_per_epoch=spe, total_steps=total_steps, warmup_steps=warmup, save_freq=int(save_freq),
        expected_save_steps=_saves, upload_steps=_uploads, upload_full_steps=_full, upload_policy=upload_policy,
        gpu_profile=GPU_PROFILE, gpu_profile_deviation=GPU_PROFILE_DEVIATION,
        overrides=ov, changes_vs_official=changes, env=env, official_script=OFFICIAL["path"], module=OFFICIAL["module"],
        fork_commit=FORK_COMMIT, upstream_reference_commit=UPSTREAM_REFERENCE_COMMIT, keep_official_env_vars=KEEP_OFFICIAL_ENV_VARS,
        resume_from_path=resume_from_path, total_training_steps_override=total_training_steps, note=extra_note,
        train_file=train_path, train_file_sha256=train_sha, training_view=view, logged_loss_scale=view["logged_loss_scale"],
        official_world_size=OFFICIAL_WORLD_SIZE, residual_differences=residual,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    )
    return spec


def print_run_spec(spec):
    print(f"run_id={spec['run_id']} kind={spec['kind']} lr={spec['lr_str']} epochs={spec['epochs']} lr_scheduler={spec.get('lr_scheduler', 'cosine')} n_gpus={spec['n_gpus']}"
          f" | 論文の条件: {spec.get('paper_condition') or '該当なし（論文に無い条件）'}")
    print(f"  data={spec['data_path']} rows={spec['n_rows']} sha256={spec['data_sha256'][:12]} max_length={spec['max_length']} (official {spec['official_max_length']})")
    v = spec["training_view"]
    if v["enabled"]:
        print(f"  公式 {v['official_world']} GPU の再現: ON（学習用 file={os.path.basename(spec['train_file'])} sha256={spec['train_file_sha256'][:12]}, "
              f"importance_sampling_mode=adv-only, advantage={v['advantage']}）。ログの train/loss は {v['logged_loss_scale']} 倍（表示時は公式スケールに戻す）")
    elif spec["n_gpus"] == spec["official_world_size"]:
        print(f"  GPU 台数が公式と同じ {spec['n_gpus']} 台。並べ替えと loss 倍率は不要")
    else:
        print(f"  *** 公式 {spec['official_world_size']} GPU の再現: OFF。micro batch 構成と勾配の大きさが公式と異なる ***")
    print(f"  global batch={spec['tbs']} micro/gpu={spec['micro_bsz']} grad-accum micro-batches/step={spec['grad_accum_micro_batches']}")
    print(f"  steps/epoch={spec['steps_per_epoch']} total_steps={spec['total_steps']} warmup={spec['warmup_steps']} save_freq={spec['save_freq']} -> saves at {spec['expected_save_steps']}")
    _full = spec.get("upload_full_steps", spec["upload_steps"])
    _light = [x for x in spec["upload_steps"] if x not in _full]
    print(f"  HF へ転送する step（{spec['upload_policy']}）: optimizer 状態込み（再開可）{_full}"
          + (f"、重みのみ（分析用・再開不可）{_light}" if _light else "") + "。それ以外はローカルで保存完了後に削除")
    print("  公式スクリプトとの差分:")
    for c in spec["changes_vs_official"]:
        print(f"    {c['key']}: {c['official']} -> {c['new']}   [{c['reason']}]")
    if not spec["keep_official_env_vars"]:
        print("    env: CUDA_LAUNCH_BLOCKING を外した（KEEP_OFFICIAL_ENV_VARS=False）")
    print("  公式と揃えられない残りの差（数値の丸め程度）:")
    for r in spec["residual_differences"]:
        print(f"    - {r}")


print("helpers ready: parse_official_script / build_run_spec / print_run_spec / get_base_model_local / fmt_lr / emulate_official_order / prepare_train_file")
