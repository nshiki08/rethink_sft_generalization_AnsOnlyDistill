# Answer-only (AO) 蒸留ノートブックの使い方

対象: `notebooks/ansonly_distillation.ipynb`（Google Colab 用）。
公式実装 `verl.trainer.fsdp_sft_trainer_ours` と `training_scripts/` は変更せず、ノートブックから既存の引数を渡して AO 学生を学習する。

## 1. Colab で開く

1. GitHub の Fork `nshiki08/rethink_sft_generalization_AnsOnlyDistill` で `notebooks/ansonly_distillation.ipynb` を開き、「Open in Colab」または
   `https://colab.research.google.com/github/nshiki08/rethink_sft_generalization_AnsOnlyDistill/blob/<branch>/notebooks/ansonly_distillation.ipynb` を開く。
2. ランタイム → GPU。**A100 を推奨**（公式 `requirements.txt` の torch 2.6.0 がそのまま動く）。
   - G4（RTX PRO 6000 Blackwell, sm_120）は torch 2.6.0 が動かない。セル 1 で `ALLOW_BLACKWELL_TORCH_DEVIATION=True` にすると torch 2.7.1+cu128（torch の要求で sympy も 1.13.1 → 1.13.3）を使い、公式との差として記録する
   - T4 は flash-attn 2 が動かないので不可
   - GPU メモリの推定: 1.7B 約 22 GiB、4B 約 41 GiB、8B 約 79 GiB（A100 80GB では不足、G4 96GB なら収まる見込み）
3. Colab の「シークレット」に以下を登録し、ノートブックからのアクセスを許可する。
   - `HF_TOKEN`（write 権限。checkpoint と最終モデルの保存に使う）
   - `WANDB_API_KEY`（任意。無ければ公式と同じ offline mode）

## 2. 設定

実験の設定は **設定セル**、コードの revision と環境の pin は **セル 1**（通常は変更しない）で行う。設定セルの主な項目:

| 項目 | 変数 | 既定 |
| --- | --- | --- |
| 実行モード | `RUN_MODE` | `"dry_run"`（学習も HF 書き込みもしない） / `"trial"` / `"train"` |
| run の種類 | `RUN_KIND` | `"baseline"`（公式 CoT 設定 lr 5e-5 / 8 epoch）/ `"search"` |
| 対象モデル | `MODEL_KEY` | `"Qwen3-1.7B"`（`SUPPORTED_MODELS` から選ぶ。Qwen2.5 系は理由付きで blocked） |
| 探索候補 | `SEARCH_GRID_PROPOSED`（提案 = 論文が 14B で行った 6 条件）/ `SEARCH_GRID_CONFIRMED`（確定）/ `SEARCH_RUN_LIST`（実行する `(lr, epochs, scheduler)`） | 提案のみ |
| 最大系列長 | `MAX_LENGTH_MODE` | `"auto_fit"`（全行が収まる長さ。`"official"` で 20000 に戻せる） |
| 保存周期 | `SAVE_FREQ` | 10（公式 CoT と同じ。保存は学習結果に影響しない） |
| HF へ転送する step | `HF_UPLOAD_STEPS` | `"cot_public+resume"`（論文が評価した step 10, 20, 40, 80, 160, 320, 480, 640（16 epoch は 1280 まで）と `RESUME_CKPT_EVERY` の倍数。それ以外はローカルで削除）/ `"all"` |
| 再開用の間隔 | `RESUME_CKPT_EVERY` | 80（optimizer 状態込みで送る間隔。切断時は最大この step 数を再実行） |
| 分析用 step の中身 | `HF_ANALYSIS_STEP_CONTENT` | `"model_only"`（再開用以外の step は重みだけ送る。変換・分析はできるが再開はできない）/ `"full"` |
| 公式 8 GPU の再現 | `EMULATE_OFFICIAL_WORLD_SIZE` | `True`（下記） |
| HF 保存先 | `HF_CKPT_REPO_ID` / `HF_FINAL_MODEL_REPO_ID` | `None` → HF の whoami から決める |
| 試走 | `TRIAL_NUM_ROWS`, `TRIAL_EPOCHS`, `TRIAL_SAVE_FREQ`, `TRIAL_KILL_AFTER_STEP` | 512 行 / 4 epoch / 3 / 3 |
| 再開 | `RESUME_RUN_ID`, `RESUME_STEP` | `None` / `"latest"` |
| 論文と同じ評価 | `PAPER_EVAL_ENABLED`, `PAPER_EVAL_AO_STEPS`, `PAPER_EVAL_BASE`, `PAPER_EVAL_COT_STEPS` | 無効（学習後に有効にする） |

### 公式 CoT 学習（8 GPU）との条件合わせ

公式 CoT 学生は 8 GPU で学習された（`README.md` の "We trained all models on 8 H200 GPUs."）。
公式 trainer は micro batch ごとに loss を割らずに backward するので、1 GPU 内の勾配は micro batch の和になり、GPU 間は FSDP2 が平均する。
このため GPU 台数 N が 8 と違うと、次の 2 点が公式と変わる。

- **micro batch（4 行）の組み合わせ**: loss は micro batch ごとの token 平均なので、各 token の重みが変わる
- **勾配の大きさ**: 8/N 倍になり、`clip_grad=1.0` の効き方と AdamW の eps の効き方が変わる

`EMULATE_OFFICIAL_WORLD_SIZE=True`（既定）では、公式コードを変えずに次の方法で公式 8 GPU の計算を再現する。N は 8 の約数（1, 2, 4, 8）であること。

| 方法 | 何を揃えるか |
| --- | --- |
| 学習用 parquet の行を並べ替える（`data/train_view/*.emul-w8-nN.parquet`。元の AO parquet は変更しない） | 各 step の各 micro batch が公式 8 GPU と同じ 4 行になる |
| `trainer.importance_sampling_mode=adv-only` と `advantage` 列 = N/8 | loss を N/8 倍する。2 のべき乗倍なので、勾配・clip・AdamW の入力が丸め誤差の範囲で公式 8 GPU と同じ値になる |

`optim.clip_grad` を 8/N にする方法は採らない。AdamW の eps（公式 trainer では変更できない 1e-8）の効き方が変わり、公式と一致しないため。
ログの `train/loss` は N/8 倍になるので、ノートブックの表示と実験記録では公式スケールに戻した値も出す。戻した値は、token 平均の計算精度の違い（adv-only は fp32、vanilla は bf16）で公式ログと最大 0.4% 程度ずれ得る。勾配は影響を受けない。

検証（CPU 上で公式 trainer を gloo で動かし、小型 Qwen3 と実 AO データ 512 行で 6 step）:

| 構成 | 公式 8 プロセスとのパラメータ差（公式の移動量に対する比） |
| --- | --- |
| fp32、1 GPU + 並べ替え + adv-only | 1.2e-6（丸め誤差） |
| bf16、1 GPU + 並べ替え + adv-only | 1.5〜3.6% |
| bf16、2 GPU + 並べ替え + adv-only | 0.7〜1.3% |
| bf16、1 GPU、そのまま（修正前） | 7〜14% |

bf16 で残る差は、fp32 では一致する 2 つの計算どうしでも同じ大きさで出る丸め差（勾配の加算順序）で、公式コードを変えずには消せない。他に残る差は GPU の種類、flash-attn の非決定性、padding 長による行列積の形、grad_norm ログの意味（公式は各 GPU の shard ノルムの平均）で、ドライラン（セクション 5）と実験記録に一覧が出る。

メモリ不足のとき、micro batch を 4 から変えると micro batch 構成を再現できない。条件を保つには GPU を大きくするか 2 台・4 台にする。micro batch を変える場合は `EMULATE_OFFICIAL_WORLD_SIZE=False` にし、公式との差として記録される。

## 3. 実行順（初回）

セル 0（認証）→ 1（環境構築）→ 設定 → 2 → 3 → 4-a → 4-b → 5（ドライラン）。ここまでは学習も HF への書き込みもしない（GPU の無いランタイムでは 1 GPU として確認する）。

- セル 0 と 1 は Colab のカーネル（Python 3.13）で動く。セル 1 が Python 3.12 の公式環境を作り（初回数分）、その環境で 2 つ目のカーネルを起動する。設定セル以降は先頭の `%%ao` でそのカーネルに送られる
- セル 2 は GPU があるとき、公式の `apply_fsdp2` で包んだ小型モデルの勾配と FSDP 無しの勾配を比べる（1 GPU の NCCL `ReduceOp.AVG` で勾配が壊れるという torch の未解決報告への対策）。一致しなければ学習セルは止まる

- 3 で `AO_DATA_READY`、4-b で `MASK_CHECK_OK` が `True` にならなければ、本学習のセルは開始しない。
- `EMULATE_OFFICIAL_WORLD_SIZE=True` では、4-b の `VIEW_CHECK_OK`（公式 8 GPU の micro batch 構成の再現確認）も必要。4-b は学習に使うのと同じ GPU 台数で確認するので、学習する GPU ランタイムで 4-b をもう一度実行する（台数が違うと本学習のセルが止まる）。`False` にした場合は警告を出して学習し、公式との差として記録する。
- 5 で起動コマンド・想定 step 数・保存回数・HF 保存先・衝突の有無を確認する。

次に `RUN_MODE="trial"` にして 6-a → 6-b（試走）。試走の流れ:
学習 → `TRIAL_KILL_AFTER_STEP` の checkpoint 保存完了を検知 → 学習プロセスを一時停止（SIGSTOP）→ HF へ転送・検証 → プロセス終了（切断の模擬）
→ ローカル checkpoint を削除 → HF から取得・manifest 検証 → 新プロセスで再開 → 最終 step まで学習 → 中断なしの参照 run と lr / loss を比較。
試走の run は `trial-` で始まり、本学習と別ディレクトリ・別データ（先頭 512 行、4 epoch = 8 step）を使う。試走の checkpoint は本学習に使わない。
試走 run の再開は同じセッション内でだけ成立する（試走データは 6-b が作るため）。

本学習は `RUN_MODE="train"`、`RUN_KIND="baseline"` で 7、`RUN_KIND="search"` かつ `SEARCH_RUN_LIST` を設定して 8。

評価（セクション 9）は論文と同じ方法: 公式の `evaluation/math_eval/math_eval_budget.py` を無変更で実行する（MATH500 avg@3、AIME24 avg@10、temperature 0.6、最大 32768 token、math-verify）。
学習が終わってから `PAPER_EVAL_ENABLED=True` にする。初回は評価用の環境（Python 3.12 + vLLM 0.8.5）を作る。論文と同じく各 step（10〜640）の推移を評価し、同じ step の公開 CoT 学生の論文値を並べて表示する。
G4 では vLLM 0.8.5 が動かないので、評価は A100 のランタイムで行う。論文との対応の一覧は `docs/ansonly/paper_alignment.md`。
学習中は 1 セルが終了まで動き続け、監視スレッドが保存完了ごとに checkpoint を HF の
`runs/<run_id>/global_step_<N>/` へ転送・検証し、完了マーカー `ao_upload_verified.json` を置く。過去 step は上書き・削除しない。

## 4. 切断後の再開

1. 新しいランタイム（保存時と同じ種類の GPU）でセル 0 → 1 → 設定 → 2 → 3 → 4-a → 4-b → 6-a を実行する（同じ Fork commit・同じデータ・同じ Base revision が復元される）。
2. 設定セルで `RUN_MODE="train"`、`RESUME_RUN_ID="<run_id>"`（`RESUME_STEP` は `"latest"` か整数）を設定する。
3. 「再開」セルを実行する。HF の checkpoint 一覧を表示し、optimizer 状態込みで転送・検証済みの step（重みだけの step は候補にしない）を取得して manifest（サイズ・sha256）で検証し、
   GPU 台数（FSDP world_size）・データ sha256・Base revision の互換性を確認してから
   `trainer.resume_mode=resume_path trainer.resume_from_path=...` で再開する。総 epoch 数・総 step 数は保存時の値を維持する。
   最後の保存以降の step は再実行になる。

## 5. 成果物

- ローカル: `/content/ao_work/{data,models,ckpt,log,records,merged}`（Colab のディスク。切断で消える）
- HF checkpoint repo: `runs/<run_id>/global_step_<N>/`（model/optimizer/extra の shard, `data.pt`, `huggingface/`, `ao_ckpt_manifest.json`, `ao_upload_verified.json`。重みだけの step は optimizer の shard なし）と `runs/<run_id>/experiment_record_*.json`
- HF の使用量（実測サイズから計算）: 1.7B は再開用 1 個 10.34 GB・重みのみ 3.46 GB で baseline 1 run 約 93 GB、4B は 24.15 GB・8.06 GB で約 217 GB。HF PRO の非公開枠は 1 TB（超過分は 1 TB あたり月 $18）
- HF 最終モデル repo: `step<N>/`（`verl.model_merger` で変換した HF 形式）と `README.md`（Model Card）

## 6. 未検証の項目

このノートブックは GPU の無い環境で検証した（構文、データ抽出・監査の全行実行、tokenizer での長さ計算、公式 dataset クラスによる全行の loss mask 検証、
公式 trainer の import、ドライラン、および公式 trainer の代わりに保存形式だけを模した偽プロセスを使った 起動・監視・一時停止/終了・再開・manifest・記録 の処理）。
以下は Colab で各セルを実行し、ログを確認するまで「動作確認済み」ではない。

- Colab 上での公式環境の作成（uv の Python 3.12、公式 pin、flash-attn wheel）と公式環境カーネルの起動。同じ手順は CPU で検証済み（`tests/test_env_bootstrap.py`）
- G4 での torch 2.7.1+cu128 と flash-attn の動作、1 GPU の FSDP2 勾配確認の GPU 上での結果
- 試走（学習・保存・HF 転送・終了・再開）、本学習、GPU メモリ、所要時間
- `verl.model_merger` による変換と最終モデルの読み込み・生成
