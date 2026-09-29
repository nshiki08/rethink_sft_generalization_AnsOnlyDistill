# 未決定事項と未検証項目

## 未決定

| 項目 | 状態 |
| --- | --- |
| dev set（LR / epoch の選択に使う独立データ） | 未確定。学習データ（Math-CoT-20k / OpenR1 由来の同一問題）とも最終テストとも重ならないデータが必要 |
| 探索範囲 | 提案 `lr ∈ {1e-5, 2e-5, 5e-5, 1e-4}`, `epochs ∈ {1, 2, 4, 8}`。教授との合意待ち（`SEARCH_GRID_CONFIRMED=None`） |
| HF の容量 | checkpoint 1 個 ≈ bf16 重み × 3（AdamW 2 状態込み）。1.7B で約 10 GB、1 run 18 回転送で約 180 GB。削減方法を検討中 |

## 抽出で参照正解と一致しなかった 3 行

行 12364, 16249, 16611。いずれも答えが方程式で、教師の式と参照正解の式は同値だが、math-verify の比較が失敗する。教師の原文を AO target として残す（`AO_ACKNOWLEDGED_VERIFY_MISMATCH_ROWS`）。

## 未検証（GPU が必要）

- Colab 上での依存関係の導入と flash-attn wheel
- 試走（学習・保存・HF 転送・終了・再開）、本学習、GPU メモリ、所要時間
- `verl.model_merger` による変換と最終モデルの読み込み・生成
