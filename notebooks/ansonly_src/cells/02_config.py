# @title 設定セル（編集はここだけ）
import os, sys, json, re, time, datetime, pathlib

# =============================================================================
# 実行モード
#   dry_run : データ・設定・コマンド・想定 step 数を確認するだけ。学習も HF 書き込みも行わない（既定）
#   trial   : 少数 step の学習で、メモリ・保存・HF 転送・プロセス終了→再開を確認する
#   train   : baseline または探索候補を所定の条件で学習する
# =============================================================================
RUN_MODE = "dry_run"          # "dry_run" | "trial" | "train"
RUN_KIND = "baseline"         # train 時に "baseline" | "search"
RUN_ID_SUFFIX = ""            # 同名 run と衝突する場合に付ける識別子（例 "-v2"）。空なら付けない

# =============================================================================
# 対象モデル（まず Qwen3 系列。サイズはここで選ぶ）
#   1.7B: 小規模な動作確認候補 / 4B: 本実験候補（Colab で実行可能かは試走で実測する）
# =============================================================================
MODEL_KEY = "Qwen3-1.7B"

SUPPORTED_MODELS = {
    # revision は 2026-09-24 に HF API で確認した commit sha。ノートブック実行時にも再記録する
    "Qwen3-1.7B": dict(status="supported",
                       base_repo="Qwen/Qwen3-1.7B-Base", base_revision="ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
                       cot_repo="jasonrqh/Qwen3-1.7B_Math-CoT-20k_lr5e-5_ep8_bs256", cot_revision="e4fcd58c7b104d3444e0e1d1b470f5107fcc2618",
                       cot_subfolder="step640", official_script="Qwen3-1.7B_Math-CoT-20k_lr5e-5_ep8_bs256.sh", note="小規模な動作確認候補"),
    "Qwen3-4B": dict(status="supported",
                     base_repo="Qwen/Qwen3-4B-Base", base_revision="906bfd4b4dc7f14ee4320094d8b41684abff8539",
                     cot_repo="jasonrqh/Qwen3-4B_Math-CoT-20k_lr5e-5_ep8_bs256", cot_revision="aaf1c51a6a2ee319750cafbd7cb8be0df10018b8",
                     cot_subfolder="step640", official_script="Qwen3-4B_Math-CoT-20k_lr5e-5_ep8_bs256.sh", note="本実験候補（Colab で収まるかは要実測）"),
    "Qwen3-8B": dict(status="supported",
                     base_repo="Qwen/Qwen3-8B-Base", base_revision="49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
                     cot_repo="jasonrqh/Qwen3-8B_Math-CoT-20k_lr5e-5_ep8_bs256", cot_revision="522decd73c0b46f7029c2a315b3cc177c23a3498",
                     cot_subfolder="step640", official_script="Qwen3-8B_Math-CoT-20k_lr5e-5_ep8_bs256.sh", note="Colab 単一 GPU では full SFT が収まらない可能性が高い"),
    "Qwen3-14B": dict(status="supported",
                      base_repo="Qwen/Qwen3-14B-Base", base_revision="0b0bd3732e2c374d483664439ea334928b65f304",
                      cot_repo="jasonrqh/Qwen3-14B_Math-CoT-20k_lr5e-5_ep8_bs256", cot_revision="e8e6542249d647ac82c556a93410a9c0cafd41cb",
                      cot_subfolder="step640", official_script="Qwen3-14B_Math-CoT-20k_lr5e-5_ep8_bs256.sh", note="Colab 単一 GPU では full SFT が収まらない可能性が高い"),
    # Qwen2.5 系列: 公式スクリプトは Qwen/Qwen2.5-{size}-no-sys-prompt を指定する（HF API は 401 = 非公開または存在しない）。
    # 公開 Base の chat template は system message が無いと "You are a helpful assistant." を自動挿入するため tokenization が公式と一致しない。
    # 関係・準備方法を確認するまで黙って置き換えない → blocked。
    "Qwen2.5-1.5B": dict(status="blocked", base_repo="Qwen/Qwen2.5-1.5B", base_revision="8faed761d45a263340a0528343f099c05c9a4323",
                         cot_repo="jasonrqh/Qwen2.5-1.5B_Math-CoT-20k_lr5e-5_ep8_bs256", cot_revision="dd3c28b7846607e026f4ecc48eeb20e8d48035f1",
                         cot_subfolder=None, official_script="Qwen2.5-1.5B_Math-CoT-20k_lr5e-5_ep8_bs256.sh",
                         note="公式は Qwen/Qwen2.5-1.5B-no-sys-prompt（非公開）。公開 Base との差（chat template の既定 system prompt）が未解決"),
    "InternLM2.5-20B": dict(status="blocked", base_repo="internlm/internlm2_5-20b", base_revision="dc786ab20a7fd882ccef61df28e13a940edca24f",
                            cot_repo="jasonrqh/InternLM2.5-20B_Math-CoT-20k_lr5e-5_ep8_bs256", cot_revision="fa6f45a03525bb5a52046e3217f4ea5082aa4c75",
                            cot_subfolder=None, official_script="InternLM2.5-20B_Math-CoT-20k_lr5e-5_ep8_bs256.sh",
                            note="今回の対象外（trust_remote_code モデル。公式 micro_bsz=2）"),
}
TEACHER = dict(repo="Qwen/Qwen3-32B", revision="9216db5781bf21249d130ec9da846c4624c16137")  # 記録用。ダウンロードしない

# =============================================================================
# コードとデータの revision
# =============================================================================
FORK_REPO_URL = "https://github.com/nshiki08/rethink_sft_generalization_AnsOnlyDistill"
FORK_REF = "main"                 # ブランチ名または commit。実行時に HEAD の commit を記録する
UPSTREAM_REPO_URL = "https://github.com/Nebularaid2000/rethink_sft_generalization"   # 参照専用（push 無効化）
UPSTREAM_REFERENCE_COMMIT = "71a442ea8f0adc4a1df4529d3c43393ac6e504fd"               # 再現用の参照 commit

DATASET_REPO = "jasonrqh/Math-CoT-20k"
DATASET_REVISION = "1435fb21d4fecc8ad4966a26f22a874cf2b527f1"
DATASET_FILE = "Math-CoT-20k.parquet"
DATASET_EXPECTED_SHA256 = "3166f342ec838ebe0979be43923abb33fa1c80f33043aa6acc04e5d82161142a"
DATASET_EXPECTED_ROWS = 20480
DATASET_ORIGIN_POOL = "open-r1/OpenR1-Math-220k"   # 元問題プール（記録のみ。DeepSeek-R1 出力は抽出元にしない）

# =============================================================================
# AO データの抽出
#   teacher_answer 列 = response（教師 Qwen3-32B の CoT+回答）から抽出した最終回答
#   last_boxed_verbatim（既定）: 最後の \boxed{...} を教師の原文のまま保持（CoT・説明文は含めない）
#   boxed_content            : \boxed{} の中身のみ（教師原文の形を崩すので既定にしない）
#   last_boxed_line          : 最後の \boxed{} を含む原文の行（説明文が混ざり得るので既定にしない）
# =============================================================================
AO_TARGET_STYLE = "last_boxed_verbatim"
# math-verify で参照正解 (answer 列) と一致しなかった行のうち、内容を目視確認して「教師の最終回答として保持する」と決めた行番号。
# 2026-09-24 の監査では 3 行。いずれも同値な方程式の別表記（例 x^2/5 + 3y^2/5 = 1 と x^2+3y^2=5）で math-verify の関係式比較が失敗する。
# 実際の不一致集合がこのリストと異なれば本学習は開始しない（未解決の抽出問題として扱う）。
AO_ACKNOWLEDGED_VERIFY_MISMATCH_ROWS = [12364, 16249, 16611]

# =============================================================================
# 学習設定
#   公式基準（公開 CoT スクリプト）: AdamW β=(0.9,0.999), lr 5e-5, 8 epoch, global batch 256, cosine + warmup 10%,
#   weight decay 0.01, grad clip 1.0, bf16, max_length 20000, truncation right, shuffle False, token-mean, grad ckpt, FSDP
#   → 実際の値は公式スクリプトと sft_trainer.yaml を実行時に読み取って表示する（ここに書いた値は目安）
# =============================================================================
OFFICIAL_BASELINE = {"lr": 5e-5, "epochs": 8, "status": "confirmed: 公開 CoT スクリプトの値"}

# 探索対象は optim.lr と trainer.total_epochs。提案値と確定値を分ける。
SEARCH_GRID_PROPOSED = {"lr": [1e-5, 2e-5, 5e-5, 1e-4], "epochs": [1, 2, 4, 8]}   # 提案（未確定）
SEARCH_GRID_CONFIRMED = None      # 教授と合意後に {"lr": [...], "epochs": [...]} を入れる
SEARCH_RUN_LIST = []              # このセッションで実行する候補 [(lr, epochs), ...]。空なら探索セルは何もしない

# 最大系列長: "auto_fit" = 全行が切り詰められずに収まる最小の長さ（MAX_LENGTH_ROUND_TO の倍数に切り上げ）
#             "official" = 公式の 20000 / 整数 = 明示指定
MAX_LENGTH_MODE = "auto_fit"
MAX_LENGTH_ROUND_TO = 64

# =============================================================================
# 公式 CoT 学生の学習条件（8 GPU）への合わせ込み
#   公式 CoT 学生は 8 GPU で学習された（README.md:101 "We trained all models on 8 H200 GPUs."）。
#   公式 trainer は micro batch ごとに loss を割らずに backward するので、1 GPU 内の勾配は micro batch の「和」になる
#   （verl/trainer/fsdp_sft_trainer_ours.py:467。:509 の /n_micro はログ用）。GPU 間は FSDP2 が「平均」する（ReduceOp.AVG）。
#   そのため GPU 台数 N が 8 と違うと、次の 2 点が公式と変わる:
#     (1) 1 micro batch（4 行）に入る行の組み合わせ。loss は micro batch ごとの token 平均なので、token の重みが変わる
#     (2) 勾配の大きさが 8/N 倍になり、clip_grad=1.0 の効き方と AdamW の eps の効き方が変わる
#   EMULATE_OFFICIAL_WORLD_SIZE=True のとき、公式コードを変えずに次の 2 点で公式 8 GPU の計算を再現する:
#     - 学習用 parquet の行を並べ替え、各 step の各 micro batch を公式 8 GPU と同じ 4 行にする（元の AO parquet は変更しない）
#     - trainer.importance_sampling_mode=adv-only と advantage 列 = N/8 で loss を N/8 倍する。
#       vanilla の loss を 2 のべき乗倍するだけなので、勾配・clip・AdamW の入力は公式 8 GPU と同じ値になる（丸め誤差の範囲。
#       CPU 上の公式 trainer で fp32 なら移動量比 1e-6 で一致、bf16 では加算順序の違いで数 % の丸め差）。ログの train/loss は N/8 倍になる
#   N は 8 の約数（1, 2, 4, 8）であること。N=8 のときは何もしない（公式と同じ）。
# =============================================================================
OFFICIAL_WORLD_SIZE = 8
EMULATE_OFFICIAL_WORLD_SIZE = True

MICRO_BATCH_OVERRIDE = None       # 公式の micro batch 4 を変えると (1) を再現できない。EMULATE_OFFICIAL_WORLD_SIZE=True では None 以外を禁止。
                                  # メモリ不足のときは、条件を保ったまま GPU を大きくするか 2/4 台にする。micro batch を変える場合は
                                  # EMULATE_OFFICIAL_WORLD_SIZE=False にし、公式との差として記録される（黙って変えない）
CPU_OFFLOAD_OVERRIDE = None       # 使用不可: 公式 trainer は fsdp2 経路でも FSDP1 用の CPUOffload を渡すため FSDP2 に無視され offload されない（trainer:300-303, 337）
KEEP_OFFICIAL_ENV_VARS = True     # 公式スクリプトの export（CUDA_LAUNCH_BLOCKING=1 など）をそのまま使う。False なら CUDA_LAUNCH_BLOCKING を外し記録する

SAVE_FREQ = 10                    # trainer.save_freq（正の整数。0 は使わない）。公式 CoT と同じ 10。保存は学習結果に影響しない
# HF へ転送する step: "cot_public+resume" = 公開 CoT 学生と同じ step（10,20,40,80,160,320,480,640）と RESUME_CKPT_EVERY の倍数と最終 step。
#   それ以外の step の checkpoint は保存完了後にローカルから消す（HF に無いので再開候補にもならない）。"all" = 保存した全 step を転送
# checkpoint 1 個 ≈ bf16 params x3（AdamW 2 状態込み）: 1.7B ≈ 10 GB, 4B ≈ 24 GB。HF の容量上限に注意（ドライランで見積を表示）
HF_UPLOAD_STEPS = "cot_public+resume"
COT_PUBLIC_STEPS = [10, 20, 40, 80, 160, 320, 480, 640]   # 公開 CoT 学生の HF repo にある stepNNN（2026-09-24 に確認）
RESUME_CKPT_EVERY = 40            # 再開用に HF へ転送する間隔（step）。SAVE_FREQ の倍数にする
SEED = None                       # None = 公式既定 (trainer.seed=1 は yaml 既定。trainer 内で明示的な乱数初期化はされていない)

# =============================================================================
# Hugging Face 保存
# =============================================================================
HF_CKPT_REPO_ID = None            # None → f"{HF_ACCOUNT_NAME}/rethink-sft-ao-checkpoints"（HF の whoami から決める）
HF_CKPT_PRIVATE = True
HF_UPLOAD_CHECKPOINTS = True      # 学習中に保存完了した checkpoint を監視スレッドが HF へ転送する
CKPT_MANIFEST_SHA256 = True       # manifest に sha256 を記録し、転送後に HF 側の LFS sha256 と照合する
LOCAL_KEEP_LAST_N_VERIFIED_CKPTS = 2   # HF 転送・検証済みのローカル checkpoint をこの数だけ残し、古いものを削除（未検証は削除しない）
HF_FINAL_MODEL_REPO_ID = None     # None → f"{HF_ACCOUNT_NAME}/{run_id}"（推論用 HF モデル）
HF_FINAL_MODEL_PRIVATE = True

WANDB_MODE = "offline"            # 公式スクリプトと同じ offline。"online" は WANDB_API_KEY がある場合のみ
WANDB_PROJECT = "sft_generalization_ao"

# =============================================================================
# 試走（RUN_MODE="trial"）
#   AO データの先頭 TRIAL_NUM_ROWS 行だけを使い、TRIAL_EPOCHS epoch 学習する。
#   steps_per_epoch = TRIAL_NUM_ROWS // 256。TRIAL_KILL_AFTER_STEP の checkpoint が HF へ転送・検証された時点でプロセスを終了し、
#   新しいプロセスで HF から取得して再開する。試走の run_id は "trial-" で始まり、本学習と混ざらない。
# =============================================================================
TRIAL_NUM_ROWS = 512
TRIAL_EPOCHS = 4                  # 2 steps/epoch x 4 = 8 step。kill step の後に十分な step が残るようにする
TRIAL_SAVE_FREQ = 3
TRIAL_KILL_AFTER_STEP = 3         # この step の checkpoint 保存完了を検知したら学習プロセスを一時停止し、HF 転送・検証後に終了する
TRIAL_RUN_REFERENCE = True        # 中断なしの参照 run も実行し、再開後の lr / loss / データ位置を比較する

# =============================================================================
# 切断後の再開（「再開」セルで使う）
# =============================================================================
RESUME_RUN_ID = None              # 例 "Qwen3-1.7B_Math-AO-20k_lr5e-5_ep8_bs256_baseline"
RESUME_STEP = "latest"            # "latest"（最新の完了済み）または整数 step

# =============================================================================
# dev 評価（セクション 9）。dev は未確定なので既定は無効。学習データや最終 test で選ばない
# =============================================================================
DEV_EVAL_ENABLED = False
DEV_EVAL_IS_INDEPENDENT = False   # dev が学習データ (Math-CoT-20k / OpenR1 由来の同一問題) とも最終 test とも重ならないと確認したら True にする
DEV_EVAL_SOURCE = None            # jsonl のパス、または HF dataset id（例 "org/name"）
DEV_EVAL_SPLIT = "test"
DEV_EVAL_REVISION = None
DEV_EVAL_QUESTION_KEY = "problem"
DEV_EVAL_ANSWER_KEY = "answer"
DEV_EVAL_MAX_ROWS = None
DEV_EVAL_MAX_NEW_TOKENS = 512     # AO 学生は短い出力。CoT 学生や Base を評価するときは大きくする
DEV_EVAL_BATCH_SIZE = 16
DEV_EVAL_DO_SAMPLE = False        # 公式評価 (math_eval_budget.py) は temperature 0.6 / top_p 0.95 / n サンプル。dev 選択は greedy を既定にし差を記録する
DEV_EVAL_TARGETS = []             # [{"name": "...", "path_or_repo": "...", "subfolder": None, "revision": None}, ...]

# =============================================================================
# 最終モデル（セクション 10）
# =============================================================================
FINAL_RUN_ID = None               # None → このセッションで学習した run
FINAL_STEP = "last"               # "last" または整数 step

# =============================================================================
# 作業ディレクトリと環境の pin
# =============================================================================
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
WORK_DIR = "/content/ao_work" if IN_COLAB else os.path.abspath("./ao_work")
REPO_DIR = f"{WORK_DIR}/repo"
DATA_DIR = f"{WORK_DIR}/data"
MODELS_DIR = f"{WORK_DIR}/models"
CKPT_DIR = f"{WORK_DIR}/ckpt"
LOG_DIR = f"{WORK_DIR}/log"
RECORD_DIR = f"{WORK_DIR}/records"
for _d in (WORK_DIR, DATA_DIR, MODELS_DIR, CKPT_DIR, LOG_DIR, RECORD_DIR):
    os.makedirs(_d, exist_ok=True)

# 公式 requirements.txt の pin から、学習に必要な範囲を抜き出したもの（vllm/sglang/評価系は含めない）
PIP_PINNED = [
    "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
    "transformers==4.52.4", "tokenizers==0.21.4", "accelerate==1.10.1", "datasets==4.0.0",
    "tensordict==0.9.1", "torchdata==0.11.0", "peft==0.17.1",
    "hydra-core==1.3.2", "omegaconf==2.3.0", "wandb==0.21.1", "ray[default]==2.43.0",
    "codetiming==1.4.0", "dill==0.3.8", "pyarrow==21.0.0", "numpy==1.26.4",
    "math-verify==0.7.0", "latex2sympy2_extended==1.10.1", "pylatexenc==2.10",
    "huggingface_hub==0.34.4", "hf-xet==1.1.9", "safetensors==0.6.2", "einops==0.8.1", "sentencepiece==0.2.1",
    "regex==2025.7.34", "word2number==1.1",   # evaluation/math_eval/utils/parser.py（既存の評価 parser）が使う
    "Jinja2==3.1.6",                          # chat template の描画に使われ、プロンプトの token に影響し得るので公式 requirements.txt と揃える
]
FLASH_ATTN_VERSION = "2.7.4.post1"   # 公式 pin。GitHub Releases の prebuilt wheel を torch/CUDA/Python/ABI に合わせて選ぶ
EXTRA_PIP_PACKAGES = []               # import 確認で不足が出た場合に追加

# ---- 設定の妥当性 ----
assert RUN_MODE in ("dry_run", "trial", "train"), RUN_MODE
assert RUN_KIND in ("baseline", "search"), RUN_KIND
assert isinstance(SAVE_FREQ, int) and SAVE_FREQ > 0, "trainer.save_freq は正の整数。0 は使わない"
assert HF_UPLOAD_STEPS in ("all", "cot_public+resume"), HF_UPLOAD_STEPS
assert RESUME_CKPT_EVERY % SAVE_FREQ == 0 and all(s % SAVE_FREQ == 0 for s in COT_PUBLIC_STEPS) or HF_UPLOAD_STEPS == "all", \
    "RESUME_CKPT_EVERY と COT_PUBLIC_STEPS は SAVE_FREQ の倍数にする（保存されない step は転送できない）"
assert isinstance(TRIAL_SAVE_FREQ, int) and TRIAL_SAVE_FREQ > 0
assert MAX_LENGTH_MODE in ("auto_fit", "official") or isinstance(MAX_LENGTH_MODE, int)
assert AO_TARGET_STYLE in ("last_boxed_verbatim", "boxed_content", "last_boxed_line")
assert CPU_OFFLOAD_OVERRIDE is None, "CPU_OFFLOAD_OVERRIDE は fsdp2 経路で効かないので使わない（公式値 False のまま）"
assert not (EMULATE_OFFICIAL_WORLD_SIZE and MICRO_BATCH_OVERRIDE is not None), "公式 8 GPU の再現には micro batch 4 が必要。MICRO_BATCH_OVERRIDE は None にする"
assert MODEL_KEY in SUPPORTED_MODELS, f"未対応の MODEL_KEY: {MODEL_KEY}"
MODEL_INFO = SUPPORTED_MODELS[MODEL_KEY]

SESSION_STARTED_AT = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
print("RUN_MODE:", RUN_MODE, "| RUN_KIND:", RUN_KIND, "| MODEL_KEY:", MODEL_KEY, "| status:", MODEL_INFO["status"])
print("WORK_DIR:", WORK_DIR)
