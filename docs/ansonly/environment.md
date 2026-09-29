# 実行環境（Colab）

## 1. 公式環境と Colab の違い

| 項目 | 公式（`requirements.txt`） | Colab GPU ランタイム（2026-09 時点） |
| --- | --- | --- |
| Python | 3.12 以下が前提（`numpy==1.26.4`, `ray==2.43.0` の wheel は cp312 まで） | 3.13.15 |
| torch | 2.6.0（PyPI 既定は CUDA 12.4 版） | 2.11.0+cu130 |
| transformers | 4.52.4 | 5.17.0 |
| numpy | 1.26.4 | 2.1.3 |

出典: [googlecolab/backend-info](https://github.com/googlecolab/backend-info) の `os-info-gpu.txt`, `pip-freeze.gpu.txt`（2026-09-28 の commit、2026-09-29 取得）。

Colab のカーネル（Python 3.13）には公式の pin を入れられない。`import verl` は `verl/protocol.py` 経由で ray を import するので、公式コードをカーネル内で読むセルも動かない。

## 2. 方針（実装中）

- 公式の pin は Colab のカーネルとは別の Python 3.12 環境（`uv venv`）に入れる
- 環境構築セルで、その Python 上に 2 つ目の Jupyter カーネルを起動し、以降のセルのコードをそこへ送って実行する（出力・エラー・停止ボタンは中継する）
- 学習は従来どおり、そのカーネルから公式 trainer を subprocess で起動する

GPU（A100 80GB / G4）ごとの対応と確認結果は調査後に追記する。
