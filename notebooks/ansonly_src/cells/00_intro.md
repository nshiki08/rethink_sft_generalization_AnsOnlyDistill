# Answer-only (AO) 蒸留学生の学習ノートブック（Google Colab 用）

研究課題「CoT蒸留による数学能力向上は、教師のどの能力の転移によるものか」を検証するため、
公開 CoT 学生（`jasonrqh/*_Math-CoT-20k_lr5e-5_ep8_bs256`）と比較する **Answer-only (AO) 学生** を、
**公式実装 `verl.trainer.fsdp_sft_trainer_ours` を無変更のまま** 使って学習する。

| 役割 | 使用資産 |
| --- | --- |
| 教師 | `Qwen/Qwen3-32B`（記録のみ。再生成はしない） |
| 学生の学習データ | `jasonrqh/Math-CoT-20k`（20,480 行）。AO の target は `response` 列から抽出した教師の最終回答 `\boxed{...}` |
| Base / CoT 学生 | 設定セルの `SUPPORTED_MODELS` を参照（まず Qwen3 系列） |
| 学習コード | Fork `nshiki08/rethink_sft_generalization_AnsOnlyDistill`（参照元 commit `71a442ea…`） |

## 実行順

| # | セクション | 内容 | 実行するカーネル | GPU |
| --- | --- | --- | --- | --- |
| 0 | 認証 | Colab Secrets（`HF_TOKEN`, 任意で `WANDB_API_KEY`）を読む。値は表示しない | Colab | 不要 |
| 1 | 環境構築 | Fork の clone（`upstream` は push 無効）、Python 3.12 の公式環境を作り公式 pin を導入、公式環境カーネルを起動 | Colab | 不要 |
| 設定 | 設定セル | モデル・revision・保存先・実行モード・学習設定・探索候補・HF 設定をここで編集 | 公式環境 | 不要 |
| 2 | 環境記録・研究条件 | 環境の記録、HF ログイン、1 GPU の FSDP2 勾配確認、対応モデル・AO target の説明 | 公式環境 | 確認のみ |
| 3 | データ取得・AO 抽出・全行監査 | 元 20,480 行・行順・`message`・`advantage` を維持し `teacher_answer` 列を追加 | 公式環境 | 不要 |
| 4 | 長さ・target・loss mask・実効設定 | 公式 tokenizer / chat template / dataset クラスで確認。`data.max_length` の提示 | 公式環境 | 不要 |
| 5 | ドライラン（既定） | 起動コマンド・想定 step 数・保存回数・HF 保存先を確認。学習も HF 書き込みも開始しない | 公式環境 | 不要 |
| 6 | 短い試走 | 少数 step の学習 → HF 保存 → プロセス終了 → HF から取得 → 新プロセスで再開 | 公式環境 | 必要 |
| 7 | baseline | 公式 CoT 設定（lr 5e-5, 8 epoch）の AO 学習 | 公式環境 | 必要 |
| 8 | LR / epoch 探索 | 候補ごとに同じ Base から独立に学習 | 公式環境 | 必要 |
| 再開 | 切断後の再開 | HF の再開可能な checkpoint 一覧 → 選択 → 検証 → `resume_path` で再開 | 公式環境 | 必要 |
| 9 | 評価 | 論文と同じ評価（公式 `math_eval_budget.py`: MATH500 avg@3, AIME24 avg@10）を各 step で実行。dev 評価は論文に無い任意の手順 | 評価用環境（vLLM） | 必要 |
| 10 | 最終モデル | `verl.model_merger` で HF 形式へ変換、HF 保存、読み込み・生成確認、Model Card | 公式環境 | 必要 |

## 2 つのカーネル

Colab の Python は 3.13（2026-08 から）で、公式 `requirements.txt` の pin（numpy 1.26.4, ray 2.43.0 など）は Python 3.12 までしか入らない。
セル 1 が Python 3.12 の別環境（`/content/ao_work/train_env`）に公式の pin を入れ、その Python で 2 つ目のカーネルを起動する。
設定セル以降は先頭の `%%ao` によりそのカーネルで実行される（出力・エラー・停止ボタンは中継される）。Colab の「セッションを再起動」をしたらセル 0 から実行し直す。

| GPU | 環境 |
| --- | --- |
| A100（推奨） | 公式 `requirements.txt` のまま（torch 2.6.0, flash-attn 2.7.4.post1） |
| G4（RTX PRO 6000 Blackwell） | torch 2.6.0 が動かない。セル 1 の `ALLOW_BLACKWELL_TORCH_DEVIATION=True` で torch 2.7.1+cu128（と torch が要求する sympy 1.13.3）を使い、公式との差として記録する |
| T4 など（compute capability < 8.0） | flash-attn 2 が動かないので不可 |

## 公式 CoT 学習との条件合わせ（GPU 台数）

公式 CoT 学生は 8 GPU で学習された。公式 trainer は 1 GPU 内の micro batch の勾配を和で蓄積し、GPU 間は平均する。
Colab の GPU 台数 N（1, 2, 4）でそのまま動かすと、micro batch（4 行）の組み合わせと勾配の大きさ（8/N 倍）が公式と変わる。
`EMULATE_OFFICIAL_WORLD_SIZE=True`（既定）では、公式コードを変えずに次の 2 点で公式 8 GPU の計算を再現する。

- 学習用 parquet の行を並べ替え、各 step の各 micro batch を公式 8 GPU と同じ 4 行にする（元の AO parquet は変更しない）
- `trainer.importance_sampling_mode=adv-only` と `advantage=N/8` で loss を N/8 倍し、勾配を公式と同じ大きさにする（ログの `train/loss` も N/8 倍になるので、表示では公式スケールに戻す）

これで学習手順の違いは target 列だけになる。CPU 上で公式 trainer と小型モデルを使った検証では、fp32 で公式 8 プロセスの run と移動量比 1e-6 で一致した。bf16（公式設定）では勾配の加算順序などによる丸め差が残る（6 step 後に 1 GPU で 1.5〜3.6%、並べ替えなしの 1 GPU は 7〜14%）。残る差の一覧はドライラン（セクション 5）と実験記録に出る。

checkpoint は公式と同じ 10 step ごとに保存する。HF へは、再開用に 80 step ごと（`RESUME_CKPT_EVERY`）と最終 step を optimizer 状態込みで、公開 CoT 学生と同じ残りの step（10, 20, 40）を重みだけ転送する（1.7B の baseline で約 93 GB）。

## 注意

- 既定の `RUN_MODE` は `"dry_run"`。学習と HF 書き込みは `"trial"` / `"train"` にしないと始まらない。
- Colab の切断でプロセスは消える。学習中の checkpoint は監視スレッドが保存完了ごとに HF へ転送し、過去 step は消さない。
- 公式実装（`verl/`, `training_scripts/`）は変更しない。追加処理はすべてこのノートブックにある。
- `git push` 先は Fork のみ。オリジナル (`Nebularaid2000/...`) は fetch 専用で push URL を無効化する。
- このノートブックの静的検証は CPU 環境で行った。GPU を要する項目（試走・学習・再開・変換）は各セルの実行ログが確認根拠になる。未実行の項目を「動作確認済み」とは書かない。
