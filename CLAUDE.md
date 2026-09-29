# CLAUDE.md

この Fork は論文 "Rethinking SFT Generalization" の公式実装に、Answer-only (AO) 蒸留の Colab ノートブックを追加したもの。
AO 学生 = 教師 Qwen3-32B の最終回答 `\boxed{...}` だけを target にして、公開 CoT 学生と同じ条件で Base から Full-parameter SFT した学生。

## 必ず守る

- `verl/` と `training_scripts/` は変更しない（monkey patch も不可）。学習はノートブックから公式 trainer `verl.trainer.fsdp_sft_trainer_ours` を subprocess で起動し、既存の hydra 引数だけで制御する。
- AO と公式 CoT 蒸留の違いは target 列だけにする。GPU 台数などで条件が変わる場合は再現策を入れるか、差分として記録する。
- push 先は Fork `nshiki08/rethink_sft_generalization_AnsOnlyDistill` のみ。オリジナル `Nebularaid2000/rethink_sft_generalization` には push しない。
- データ本体・モデル重み・checkpoint・認証情報を commit しない。HF 上の checkpoint を自動削除・上書きしない。
- LoRA / QLoRA は使わない。

詳細と理由: [docs/ansonly/constraints.md](docs/ansonly/constraints.md)

## ノートブックの編集

`notebooks/ansonly_distillation.ipynb` は生成物。ソースは `notebooks/ansonly_src/cells/` にある。

```bash
python notebooks/ansonly_src/build_nb.py                       # ipynb を再生成
python notebooks/ansonly_src/tests/run_cells_local.py 08        # CPU でセル 1〜5 を実行
python notebooks/ansonly_src/tests/test_launch_monitor.py       # 起動・監視・再開の検証
```

## docs（必要なときに読む）

| ファイル | 内容 |
| --- | --- |
| [docs/ansonly/architecture.md](docs/ansonly/architecture.md) | セル構成、学習起動の流れ、データの流れ |
| [docs/ansonly/official_8gpu_emulation.md](docs/ansonly/official_8gpu_emulation.md) | 公式 8 GPU を 1 GPU で再現する方法（行の並べ替え + adv-only）、不採用案、残る差、検証結果 |
| [docs/ansonly/environment.md](docs/ansonly/environment.md) | Colab の Python / GPU / 依存関係の扱い |
| [docs/ansonly/pitfalls.md](docs/ansonly/pitfalls.md) | ハマりポイント |
| [docs/ansonly/open_items.md](docs/ansonly/open_items.md) | 未決定事項（dev set、探索範囲、HF 容量）と未検証項目 |
