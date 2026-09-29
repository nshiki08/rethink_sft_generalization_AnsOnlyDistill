# 論文の設定との対応

出典: 論文 arXiv 2604.06628 v2（2026-08-15、COLM 2026 camera-ready）。公式コードは参照 commit `71a442e`。
方針: 論文と変えてよいのは **AO の target** と **ハイパラ探索** だけ。それ以外は論文と同じにし、揃えられないものは理由と影響を記録する。

## 1. 論文に dev はあるか

**無い。** 論文はハイパラも checkpoint も選択していない（本文・付録に dev / validation / held-out / early stopping の記述なし）。
- 既定の設定（lr 5e-5, 8 epoch など）は固定値として使い、Sec. 3 で条件を変えて比較している（Sec. 2.1）
- 結果は test ベンチマーク上の **step ごとの推移**（step 10〜640、16 epoch は 1280 まで。App. D）と、要約表の **最終 step**（Tab. 2 "We report the performance of the last checkpoint"）
- 評価スクリプトに `humanoid_data_1k_fortest.jsonl`（1k の held-out らしきファイル）の設定があるが、どのスクリプトからも使われず、公開もされていない

ノートブックの 9-b（dev 評価）は論文に無い追加手順で、既定は無効。探索候補を 1 つに選ぶ必要がある場合だけ使う。

## 2. 学習

| 項目 | 論文 | ノートブック | 差と理由 |
| --- | --- | --- | --- |
| 学習データ | Math-CoT-20k（20,480 行） | 同じ行・行順・prompt（`message` 列） | なし |
| target | 教師の CoT + まとめ + 回答 | 教師の最後の `\boxed{...}` のみ | **AO（許容した差）** |
| Base | Qwen3-1.7B/4B/8B/14B-Base | 同じ（revision 固定） | なし |
| optimizer / lr / batch / schedule / warmup / weight decay / epoch | AdamW, 5e-5, 256, cosine, 10%, 0.01, 8（Tab. 3, App. B.2） | 公式スクリプトの値をそのまま使う | なし（baseline） |
| 学習コード | verl SFT trainer | 公式 `fsdp_sft_trainer_ours` を無変更 | なし |
| GPU | 8 × H200 | 1 × A100（など） | 行の並べ替えと adv-only で micro batch 構成と勾配の大きさを公式と同じにした。bf16 の加算順序などの丸め差が残る（[official_8gpu_emulation.md](official_8gpu_emulation.md)） |
| `data.max_length` | 20000（Tab. 3 は prompt 3072 + response 16384） | 1536（全行が収まる長さ） | padding の量だけ。loss と勾配は数学的に同じ（行列積の形による丸め差のみ）。20000 では logits (4, 20000, 151936) bf16 の 3 コピーだけで約 73 GB になり 1 GPU に載らない |
| checkpoint の保存周期 | 10 step | 10 step | なし |
| checkpoint の保存内容 | 重みのみ（`save_contents=[model]`） | 重み + optimizer + extra | Colab 切断からの再開のため。学習結果には影響しない |
| 環境 | requirements.txt（torch 2.6.0 など） | 同じ pin（A100）。G4 は torch 2.7.1 | G4 のみ差（[environment.md](environment.md)） |
| seed | 既定（主実験は 1 回） | 既定、1 回 | なし |

## 3. ハイパラ探索（許容した差）

論文が lr / epoch / scheduler を変えたのは **Qwen3-14B-Base のみ**（1.7B/4B/8B は既定のみ。公開 CoT 学生も既定のみ）。

| 論文の条件 | 出典 | ノートブックの提案 |
| --- | --- | --- |
| lr 5e-5, 8 epoch, cosine | 既定（Sec. 2.1） | baseline |
| lr 5e-5, 1 epoch | Tab. 4 | 探索候補 |
| lr 1e-5, 1 epoch / 2 epoch | Tab. 4 | 探索候補 |
| lr 5e-5, 16 epoch, cosine | Sec. 3.4 Setting 2 | 探索候補 |
| lr 5e-5, 16 epoch, constant | Sec. 3.4 Setting 3 | 探索候補 |
| lr 1e-4, 16 epoch, constant | Sec. 3.4 Setting 4 | 探索候補 |

- 提案は `SEARCH_GRID_PROPOSED`（未確定）。確定は `SEARCH_GRID_CONFIRMED`
- 1.7B/4B ではこれらの条件の公開 CoT 学生が無い。CoT との比較が同じ条件でできるのは baseline だけ
- 探索の結果の扱い（論文と同じく全条件の推移を並べるか、dev で 1 つ選ぶか）は未決定（[open_items.md](open_items.md)）

## 4. 評価

| 項目 | 論文 | ノートブック（セクション 9-a） | 差 |
| --- | --- | --- | --- |
| 数学ベンチマーク | MATH500, AIME24 のみ | 同じ | なし |
| 実行コード | `evaluation/math_eval/math_eval_budget.py`（vLLM） | 同じスクリプトを無変更で実行（データの絶対パスはシンボリックリンクで解決） | なし |
| decoding | temperature 0.6, 最大 32768 token（Sec. 2.2）。コード上 top_p 0.95, seed 1234, stop = eos | 同じ（スクリプトの値） | なし |
| 指標 | MATH500 avg@3, AIME24 avg@10（App. B.4） | 同じ（スクリプトの `average_pass_rate`） | なし |
| prompt | chat template、system prompt なし（App. B.5） | 同じ（スクリプトの `build_prompt`。学習の prompt と違い `\n` の後に空白が 1 つ入る点も公式どおり） | なし |
| 採点 | math-verify | 同じ | なし |
| GPU | 2 × H200（tensor parallel 2） | 1 GPU（tensor parallel 1） | 数値がわずかに変わると temperature 0.6 のサンプリング結果も変わる。論文値とはサンプリングの揺らぎの範囲（AIME24 は 30 問なので数ポイント）で異なり得る。CoT と厳密に比べるなら CoT も同じ環境で評価する（`PAPER_EVAL_COT_STEPS`） |
| vLLM | 0.8.5（requirements.txt）、XFORMERS backend | 同じ。学習用とは別の Python 3.12 環境に入れ、間接的な依存も requirements.txt の版に制約する（矛盾する protobuf / typer を除く） | protobuf と typer の版。G4 は vLLM 0.8.5 非対応のため評価は A100 で行う |
| 評価する step | 10, 20, 40, 80, 160, 320, 480, 640（16 epoch は 1280 まで） | 同じ（HF に送る step もこれに合わせた） | なし |
| 数学以外 | LiveCodeBench v2, GPQA-Diamond, MMLU-Pro, IFEval, AlpacaEval, HaluEval, TruthfulQA, HEx-PHI | 未実装 | 公式リポジトリに MMLU-Pro / HaluEval / TruthfulQA / LiveCodeBench のデータが無く、判定モデルの場所も作者の環境に固定されている。GPQA-Diamond と IFEval はデータがあり追加可能 |

論文の値（App. D Table 18〜24、1.7B/4B の Base・CoT・NoCoT の MATH500/AIME24）は設定セルの `PAPER_REFERENCE` にあり、9-a の表に同じ step の CoT の値を並べて表示する。
