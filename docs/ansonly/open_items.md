# 未決定事項と未検証項目

## 1. 探索結果の扱い: 決定済み（論文と同じ）

論文には dev が無く、条件や checkpoint の選択もしていない（[paper_alignment.md](paper_alignment.md)）。
方針: 論文に無い手順は行わない。探索の各条件を、論文と同じ評価（セクション 9）で step ごとに評価して並べて報告する。1 つを選ぶ処理は無い。

## 2. 探索範囲: 未確定

提案は論文が Qwen3-14B で行った最適化条件（[paper_alignment.md](paper_alignment.md) の 3 節）: lr 5e-5 / 1 epoch、lr 1e-5 / 1・2 epoch、lr 5e-5 / 16 epoch（cosine・constant）、lr 1e-4 / 16 epoch（constant）。
教授との合意待ち（`SEARCH_GRID_CONFIRMED=None`）。1.7B/4B ではこれらの条件の公開 CoT 学生が無い。

## 3. HF の容量

実測サイズ（公式 trainer と同じ FSDP2・bf16・AdamW の checkpoint を CPU で作成して計測）:

| | 1.7B | 4B | 8B（推定） |
| --- | --- | --- | --- |
| 再開用（重み + optimizer 状態 + extra + data.pt + huggingface/） | 10.34 GB | 24.15 GB | 約 49.2 GB |
| うち optimizer 状態（exp_avg, exp_avg_sq, bf16） | 6.88 GB | 16.09 GB | 約 32.8 GB |
| 重みのみ | 3.46 GB | 8.06 GB | 約 16.4 GB |

HF の上限（[Storage limits](https://huggingface.co/docs/hub/storage-limits), 2026-09-29 取得）: PRO の非公開は 1 TB まで、超過は 1 TB あたり月 $18。公開は「最大 10 TB（best-effort）」。
PRO の容量は論理サイズで数える（Xet の重複排除による課金は Enterprise のみ）。optimizer 状態は毎 step 全体が変わるので重複排除も効かない。

| 転送方針 | 1.7B baseline（640 step） | 4B baseline |
| --- | --- | --- |
| 旧既定: 40 step ごと + 論文の評価 step、すべて再開用 | 186 GB | 435 GB |
| **現既定: 80 step ごとと最終 step を再開用、論文の評価 step の残り（10, 20, 40）は重みのみ** | **93 GB** | **217 GB** |
| 160 step ごとを再開用、残りは重みのみ | 55 GB | 129 GB |

現既定で、baseline と論文の 6 条件（[paper_alignment.md](paper_alignment.md) の 3 節）をすべて学習した場合:

| 条件 | step 数 | 1.7B | 4B |
| --- | --- | --- | --- |
| baseline（8 epoch） | 640 | 93 GB | 217 GB |
| 1 epoch × 2 条件 | 80 | 21 GB × 2 | 48 GB × 2 |
| 2 epoch | 160 | 31 GB | 72 GB |
| 16 epoch × 3 条件 | 1280 | 176 GB × 3 | 411 GB × 3 |
| 合計 | | **約 0.69 TB** | **約 1.62 TB** |

- 現既定では最初の 80 step の途中で切断すると再開できず、最初からやり直しになる（1.7B・A100 で 80 step は概算 1 時間前後: 1 step ≈ 8 × 1.72e9 × 256 × 1536 FLOP ≈ 5.4e15、実効 125 TFLOPS と仮定。試走で実測する）
- 4B で全条件を学習すると現既定でも 1 TB を超える（16 epoch の 3 条件が大半）。選択肢: `RESUME_CKPT_EVERY=160`、run 終了後に不要な optimizer 状態を HF から消す（ノートブックは自動削除しない。削除は利用者が決める。Git LFS/Xet のファイル削除まで行わないと容量は戻らない）、有料枠
- HF の Storage Buckets（上書き可能）は `huggingface_hub>=1.5` が必要で、公式 pin（0.34.4）と両立しない

## 4. 抽出で参照正解と一致しなかった 3 行（解決済み: 教師の原文を残す）

3 行とも答えが方程式で、教師の式は参照正解の式の定数倍（1/5, 1/6, 5 倍）。sympy で同じ曲線であることを確認し、問題も独立に解いて教師の答えが正しいことを確認した。

| 行 | 問題（要約） | 参照正解 | 教師の最後の `\boxed{}` | 関係 |
| --- | --- | --- | --- | --- |
| 12364 | 条件を満たす楕円の方程式 | `x^{2}+3y^{2}=5` | `\dfrac{x^2}{5} + \dfrac{3y^2}{5} = 1` | 教師 = 参照 ÷ 5 |
| 16249 | 3 点を通る円の方程式 | `6x^{2}+6y^{2}-9x-14y-2=0` | `\left(x - \dfrac{3}{4}\right)^2 + \left(y - \dfrac{7}{6}\right)^2 = \dfrac{325}{144}` | 教師 = 参照 ÷ 6 |
| 16611 | 点 Q の軌跡の方程式 | `\frac{(x-1)^{2}}{\frac{5}{2}}+\frac{(y-1)^{2}}{\frac{5}{3}}=1` | `2(x - 1)^2 + 3(y - 1)^2 = 5` | 教師 = 参照 × 5 |

math-verify 0.7.0 が失敗する理由（`math_verify/grader.py`）:
1. 方程式の比較は「左辺 − 右辺」が一致するかを見る（:331-335）。移項だけなら一致するが、定数倍は一致しない
2. 代替の `solve` 比較（:271-289）は解が dict のリストになり、`sorted()` が `TypeError` で失敗する（math-verify の不具合）。例外は捕捉されて「不一致」になる

データ全体で `=` を含む答えは 896 行、うち不一致はこの 3 行だけ（移項の違いは 62 行で正しく一致と判定されている）。

## 5. 未検証（GPU が必要）

- Colab 上での公式環境の作成（CUDA 版 torch）、flash-attn、1 GPU の FSDP2 勾配確認
- 試走（学習・保存・HF 転送・終了・再開）、本学習、GPU メモリ、所要時間
- `verl.model_merger` による変換と最終モデルの読み込み・生成
- 重みのみの checkpoint の HF 転送と、そこからの変換
- 論文と同じ評価（セクション 9）: 評価用環境（vLLM 0.8.5）の作成、生成、所要時間。CPU ではデータの読み込み、結果ファイルの読み取り、採点だけ確認した
- 数学以外の論文の評価（GPQA-Diamond, IFEval などは未実装）
