# CLAUDE.md

この Fork は論文 "Rethinking SFT Generalization" の公式実装に、Answer-only (AO) 蒸留の Colab ノートブックを追加したもの。
AO 学生 = 教師 Qwen3-32B の最終回答 `\boxed{...}` だけを target にして、公開 CoT 学生と同じ条件で Base から Full-parameter SFT した学生。
研究目的: CoT 蒸留学生と AO 蒸留学生を論文と同じ MATH500 / AIME24 で比べ、数学能力の向上に CoT が寄与しているかを調べる。論文の数学以外の評価は対象外。

## 必ず守る

- `verl/` と `training_scripts/` は変更しない（monkey patch も不可）。学習はノートブックから公式 trainer `verl.trainer.fsdp_sft_trainer_ours` を subprocess で起動し、既存の hydra 引数だけで制御する。
- 論文に無い手順（dev 評価、条件の選択など）は行わない。論文と変えてよいのは AO の target とハイパラ探索だけ。学習・評価のそれ以外の条件は論文に揃える（評価は公式 `evaluation/math_eval/math_eval_budget.py` を無変更で使う）。GPU 台数などで揃えられない場合は再現策を入れるか、差分として記録する。
- push 先は Fork `nshiki08/rethink_sft_generalization_AnsOnlyDistill` のみ。オリジナル `Nebularaid2000/rethink_sft_generalization` には push しない。
- データ本体・モデル重み・checkpoint・認証情報を commit しない。HF 上の checkpoint を自動削除・上書きしない。
- LoRA / QLoRA は使わない。

詳細と理由: [docs/ansonly/constraints.md](docs/ansonly/constraints.md)

## ノートブックの編集

`notebooks/ansonly_distillation.ipynb`（学習）と `notebooks/ansonly_eval.ipynb`（評価）は生成物。ソースは `notebooks/ansonly_src/cells/` にあり、評価ノートブックは同じセルの一部（`build_nb.py` の `EVAL_CELLS`）でできている。

```bash
python notebooks/ansonly_src/build_nb.py                       # 2 つの ipynb を再生成
python notebooks/ansonly_src/tests/run_cells_local.py 08        # CPU で設定〜セル 5 を実行（公式 pin の Python）
python notebooks/ansonly_src/tests/test_launch_monitor.py       # 起動・監視・再開の検証
python notebooks/ansonly_src/tests/run_cells_local.py --eval    # 評価ノートブックのセルを CPU で実行（データ準備のセルに依存しないこと）
python notebooks/ansonly_src/tests/test_eval_helpers.py         # 評価の CPU で確認できる部分
```

Colab のカーネルは Python 3.13 で公式 pin が入らない。セル 1 が Python 3.12 の公式環境とその Jupyter カーネルを作り、設定セル以降（`%%ao`）はそこで動く。詳細は environment.md。

## docs（必要なときに読む）

| ファイル | 内容 |
| --- | --- |
| [docs/ansonly/paper_alignment.md](docs/ansonly/paper_alignment.md) | 論文の設定との対応（学習・評価・探索条件）。論文に dev は無い |
| [docs/ansonly/architecture.md](docs/ansonly/architecture.md) | セル構成、学習起動の流れ、データの流れ |
| [docs/ansonly/official_8gpu_emulation.md](docs/ansonly/official_8gpu_emulation.md) | 公式 8 GPU を 1 GPU で再現する方法（行の並べ替え + adv-only）、不採用案、残る差、検証結果 |
| [docs/ansonly/environment.md](docs/ansonly/environment.md) | Colab の Python / GPU / 依存関係の扱い |
| [docs/ansonly/pitfalls.md](docs/ansonly/pitfalls.md) | ハマりポイント |
| [docs/ansonly/open_items.md](docs/ansonly/open_items.md) | 探索（保留中）、HF 容量、未検証項目 |
