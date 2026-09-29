# 公式 8 GPU の学習を N GPU（主に 1 GPU）で再現する方法と決定事項

公式 CoT 学生は 8 GPU（H200）で学習された（`README.md` の "We trained all models on 8 H200 GPUs."）。
Colab では 1 GPU で学習するので、公式コードを変えずに 8 GPU と同じ計算になるよう入力と引数を調整する。

## 1. 公式 trainer の挙動（出典: `verl/trainer/fsdp_sft_trainer_ours.py`）

| 挙動 | 場所 | 結果 |
| --- | --- | --- |
| micro batch ごとに `loss.backward()`。`n_micro` で割らない | `:467`（`:509` の `/n_micro` はログ用） | 1 GPU 内の勾配は micro batch の**和** |
| GPU 間は FSDP2 の reduce-scatter（`ReduceOp.AVG`、fp32） | torch 2.6 `_fsdp_collectives.py:384-395, 522-530` | GPU 間は**平均** |
| loss は micro batch（4 行）ごとの token 平均 | `_compute_loss_and_backward` | どの 4 行が同じ micro batch に入るかで各 token の重みが変わる |
| `train_batch_size // world_size` 行を各 GPU が受け持つ | `:157` | 1 GPU では 256 行 = 64 micro batch |
| `DistributedSampler(shuffle=False, drop_last=True)` | dataloader 作成部 | 公式 rank r・micro t・step k の行 = `256k + r + 32t + {0, 8, 16, 24}`。毎 epoch 同じ順 |
| AdamW は `lr, betas, weight_decay` だけ渡す | `:348-353` | `eps=1e-8` 固定（hydra から変えられない） |
| 状態・勾配は bf16（`model_dtype=bf16`） | — | 勾配の累積は bf16 の加算 |

したがって N GPU でそのまま動かすと:

1. 1 step の勾配 = (1/N)·Σ(64 micro batch) となり、公式 (1/8)·Σ の **8/N 倍**
2. micro batch の 4 行の組み合わせが公式と違う

1 の結果、`clip_grad=1.0` が公式では発動しない step で発動し、AdamW の eps の効き方も変わる。
「勾配累積で 1 GPU でも 8 GPU と同じ」は、この trainer では成り立たない。

## 2. 採用した方法（`EMULATE_OFFICIAL_WORLD_SIZE=True`、N は 1, 2, 4, 8）

| 方法 | 何を揃えるか | 実装 |
| --- | --- | --- |
| 学習用 parquet の行を並べ替える | 各 step・各 micro batch の 4 行と順序を公式と同一にする | `emulate_official_order()` → `prepare_train_file()` が `data/train_view/<stem>.emul-w8-nN.parquet` を作る。元の AO parquet は変更しない |
| `trainer.importance_sampling_mode=adv-only` と `advantage` 列 = N/8 | loss を N/8 倍 → 勾配が公式と同じ大きさになる | adv-only は `losses = -advantages * log_prob`（`:1041-1046`）。N/8 は 2 のべき乗なので bf16 でも丸めずに正確に掛かる |

並べ替えの式: 公式の micro batch 一覧 `L = [(t, r) for t in 0..7 for r in 0..7]` を N GPU の (rank r', micro u) に `L[u·N + r']` として割り当てる。
`new[256k + N(4u+i) + r'] = old[256k + r + 32t + 8i]`（実装は `order[B*k + N*j + q] = B*k + r + W*(M*t + i)`）。

前提条件: 20480 % 256 == 0（端数行なし）、AO parquet が CoT と同じ行順（`ao_source_row == index`）。

## 3. 採らなかった方法と理由

| 方法 | 不採用の理由 |
| --- | --- |
| `optim.clip_grad=8/N` | clip の発動は一致するが、AdamW の更新が `8m/(8√v+eps)` となり eps の実効値が 1/8 になる。CPU 検証で公式から fp32 でも 7.6〜8.7% ずれた（並べ替え + adv-only は fp32 で 1e-6）。eps は hydra から変えられない |
| `model_dtype=fp32` | 勾配累積は fp32 になるが、optimizer も fp32 master weight になり公式（pure bf16 AdamW）から大きく変わる |
| micro batch サイズの変更 | token 平均の単位が変わる。メモリ不足時の最終手段として `EMULATE_OFFICIAL_WORLD_SIZE=False` と併用し、公式との差として記録する |
| 公式コードの改変・monkey patch | 制約で禁止 |

## 4. 残る差（`residual_differences()` がドライランと実験記録に出す）

- bf16 勾配累積の加算順序: 公式は各 GPU で 8 回、1 GPU では 64 回の bf16 加算。丸めの向きが偏らない雑音（1 step あたり相対 0.6% 程度）
- GPU の種類（公式 H200）、flash-attn backward の非決定性
- padding 長（`max_length`）による行列積の形の違い。loss と勾配には影響しない
- `train/grad_norm` のログの意味: 公式（8 GPU）は各 GPU の shard ノルムの平均、1 GPU は全体ノルム。曲線を直接比較できない。clip 自体は正しい全体ノルムで行われる
- `train/loss` は N/8 倍で記録される。ノートブックは 8/N 倍した値（`train/loss_official_scale`）も出す。adv-only は fp32、vanilla は bf16 で token 平均を取るので、戻した値は公式ログと最大 0.4% 程度ずれ得る。勾配は影響を受けない
- checkpoint のファイル名が `model_world_size_{N}_rank_{r}.pt` になる。再開は同じ N でしかできない

## 5. 検証結果（CPU、公式 trainer を gloo で実行、小型 Qwen3、実 AO データ 512 行、6 step）

| 構成 | 公式 8 プロセスとのパラメータ差（公式の移動量に対する比） |
| --- | --- |
| fp32、1 GPU + 並べ替え + adv-only | 1.2e-6（丸め誤差） |
| bf16、1 GPU + 並べ替え + adv-only | 1.5〜3.6% |
| bf16、2 GPU + 並べ替え + adv-only | 0.7〜1.3% |
| bf16、1 GPU、並べ替えなし（修正前） | 7〜14% |
| 1 GPU + 並べ替え + `clip_grad=8`（不採用案） | fp32 で 7.6〜8.7%、bf16 で 10〜16% |

検証時の注意:

- torch 2.6 の FSDP2 は CPU でそのままは動かない（`'Stream' object has no attribute 'record_event'`）。検証用 wrapper で CPU 向けの穴埋めだけ行った。リポジトリのファイルは変更していない
- gloo は `reduce_scatter_tensor(op=AVG)` を黙って SUM として扱う。CPU/gloo で素朴に 8 プロセスを回すと公式の再現にならない（補正して比較した）
- 並べ替えの正しさは、実 sampler（`DistributedSampler`）と `TensorDict.split` で「N GPU の micro batch = 公式 8 GPU の micro batch」を全 step で照合した（ノートブックのセル 4-b `VIEW_CHECK_OK` も同じ照合を行う）

## 6. 決定事項

- 既定は `EMULATE_OFFICIAL_WORLD_SIZE=True`。本学習セルは `VIEW_CHECK_OK` と `order_sha256` の一致を要求する
- `MICRO_BATCH_OVERRIDE=None`、`CPU_OFFLOAD_OVERRIDE=None` を既定とし、変えた場合は公式との差として記録する
- checkpoint は公式と同じ `SAVE_FREQ=10`。HF へ送るのは論文が評価した step（10, 20, 40, 80, 160, 320, 480, 640、16 epoch は 1280 まで）と 80 step ごと（`HF_UPLOAD_STEPS="cot_public+resume"`, `RESUME_CKPT_EVERY=80`）
