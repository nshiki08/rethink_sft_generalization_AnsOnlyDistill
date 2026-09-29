# 実行環境（Colab）

## 1. 公式環境と Colab の違い

| 項目 | 公式（`requirements.txt`） | Colab GPU ランタイム（2026-09 時点） |
| --- | --- | --- |
| Python | 3.12 以下（`numpy==1.26.4`, `ray==2.43.0` の wheel は cp312 まで） | 3.13.15（2026-08-19 から） |
| torch | 2.6.0（PyPI 既定は CUDA 12.4 版） | 2.11.0+cu130 |
| transformers | 4.52.4 | 5.17.0 |
| numpy | 1.26.4 | 2.1.3 |

出典（2026-09-29 取得）:
- [googlecolab/backend-info](https://github.com/googlecolab/backend-info): `os-info-gpu.txt`, `pip-freeze.gpu.txt`（commit `1344837`, 2026-09-28）。Python 3.13 への更新は commit `a724079`（2026-08-19）
- [googlecolab/colabtools#6081](https://github.com/googlecolab/colabtools/issues/6081): Python 3.13 への移行の告知。旧版（26.07 以前）は「ランタイムのバージョン」で選べるが、セッションをまたいで保持されない

結論: Colab のカーネルには公式の pin を入れられない。Python 3.12 のランタイム（2026.07 など）を選んでも、Colab 既定の torch 2.11 などを公式版へ入れ替える必要がある。

## 2. 採用した構成: 2 つのカーネル

```
Colab のカーネル（Python 3.13）
  セル 0: Secrets → 環境変数
  セル 1: Fork の clone → uv で Python 3.12 venv（/content/ao_work/train_env）→ 公式 pin → 確認 → 公式環境カーネルを起動、%%ao を登録
        │ jupyter_client（IPC）でコードを送り、出力・エラー・停止を中継
        ▼
公式環境カーネル（Python 3.12 + 公式 pin）
  設定セル以降（%%ao）: データ処理・公式 dataset クラスでの確認・学習起動（torchrun subprocess）・HF 転送・評価
```

- 公式コードを使う処理（`import verl` は ray を必要とする）をすべて公式環境で動かすため、学習の subprocess だけでなくセル自体を公式環境のカーネルで実行する
- 公式環境の作成は uv（`--python-preference only-managed` で python-build-standalone の 3.12 を使う。OS の Python に依存しない）。CPU での実測: venv 作成と全 pin の導入で約 20 秒（CUDA 版 torch はダウンロード量が多いので Colab では数分）
- `UV_*` 環境変数は公式環境の作成時に外す（Colab が uv 用の制約ファイルを指定している場合があり、Colab 既定の版に固定されるため）
- 公式環境カーネルには `PYTHONPATH=<repo>` だけを渡す（Colab カーネルの PYTHONPATH に入っている 3.13 用パッケージを読まないため）
- 公式環境カーネルは新しい process group で起動する。セル 1 を再実行すると、前回のカーネル（GPU メモリを持っている可能性がある）を pid ファイルから探して終了する

### pin の出典

- 公式 `requirements.txt` と同じ: torch 系、transformers, tokenizers, accelerate, datasets, tensordict, torchdata, peft, hydra-core, omegaconf, wandb, ray, codetiming, dill, pyarrow, numpy, math-verify, latex2sympy2_extended, pylatexenc, sympy, antlr4-python3-runtime, huggingface_hub, hf-xet, safetensors, einops, sentencepiece, regex, word2number, Jinja2
- 公式に pin が無いので決めたもの: `pandas==2.3.3`（`requirements.txt` に無く `setup.py` は無指定。導入時期で 3.x に変わらないよう 2.x の最終版に固定）、`ipykernel==6.29.5`（公式環境カーネル用。学習には関与しない）
- 入れないもの: vllm, sglang, xformers など学習に使わないもの

## 3. GPU ごとの対応

| GPU | compute capability | 公式 torch 2.6.0 | 対応 |
| --- | --- | --- | --- |
| A100 40GB / 80GB | 8.0 | 動く | 公式環境のまま（プロファイル `official`） |
| L4 / H100 | 8.9 / 9.0 | 動く | 同上 |
| G4 = RTX PRO 6000 Blackwell Server Edition（96 GB） | 12.0 | **動かない** | `ALLOW_BLACKWELL_TORCH_DEVIATION=True` でプロファイル `blackwell` |
| T4 | 7.5 | — | flash-attn 2 が動かないので不可 |

`blackwell` プロファイルの内容と根拠:

| 項目 | 公式 | blackwell | 根拠 |
| --- | --- | --- | --- |
| torch | 2.6.0（cu124） | 2.7.1+cu128 | torch 2.6.0 の wheel は sm_50〜sm_90 のみで PTX も無い（wheel 内の `libtorch_python.so` の arch 一覧で確認）。sm_120 を含む最初の公式版は 2.7.0 の cu128 ビルド（[PyTorch 2.7 release blog](https://pytorch.org/blog/pytorch-2-7/)） |
| flash-attn | 2.7.4.post1（torch2.6, cxx11abiFALSE） | 2.7.4.post1（torch2.7, cxx11abiTRUE） | torch2.6 用 wheel は sm_80/sm_90 のみ。torch2.7 用 wheel は sm_80/90/100/120 を含む（wheel 内の fatbin を解析） |
| 勾配の GPU 間平均・累積、clip、state_dict 読み込み | — | 2.6 と同じ | torch v2.6.0 / v2.7.0 / v2.7.1 のソース比較（`_fsdp_collectives.py`, `clip_grad.py`、verl の vendored `state_dict.py` は 2.7.0 のコピー） |
| NCCL / cuBLAS | 2.21.5 / 12.4 | 2.26.2 / 12.8 | カーネル実装が変わるので丸め誤差の範囲の差が出る |

G4 の出典: Colab の告知（[@GoogleColab, 2026-03-04](https://x.com/GoogleColab/status/2029331896409464974)、検索結果の抜粋）、compute capability は [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus)。

同じ run の途中で GPU プロファイルを変えて再開することはできない（`check_resume_compatibility` が止める）。

## 4. 1 GPU の FSDP2 勾配確認

1 GPU では FSDP2 の reduce-scatter が world size 1 の NCCL `ReduceOp.AVG` になる。torch に「world size 1 の AVG で一部の勾配が 0 になる」未解決の報告がある:
- [pytorch/pytorch#165399](https://github.com/pytorch/pytorch/issues/165399)（torch 2.8, NCCL 2.27.3）
- [pytorch/pytorch#144045](https://github.com/pytorch/pytorch/issues/144045)（torch 2.5.1）
- torch 2.8 のソースにも "NCCL ReduceOp.AVG may produce incorrect results with world size 1" という注記がある

セクション 2 で、公式の `apply_fsdp2` / `fsdp2_load_full_state_dict` で包んだ小型 Qwen3（語彙 151936、tied embedding）の勾配を、FSDP 無しの同じモデルの勾配と比べる。
全パラメータで相対差 < 1e-2 かつ「FSDP 側だけ 0 になった要素」が 0 個なら OK。`REQUIRE_FSDP2_GRAD_CHECK=True`（既定）では OK でないと学習セルが止まる。
CPU（gloo, world size 1）では差 0 を確認し、gloo が AVG を SUM として扱う 2 プロセス構成では勾配 2 倍（相対差 1.0）を検出することを確認した。GPU 上の結果は未確認。

## 5. GPU メモリの推定（公式 trainer のコードから算出、micro batch 4, max_length 1536）

| モデル | パラメータ数 | 推定ピーク | A100 80GB | G4 96GB |
| --- | --- | --- | --- | --- |
| Qwen3-1.7B-Base | 1,720,574,976 | 約 22 GiB | 収まる | 収まる |
| Qwen3-4B-Base | 4,022,468,096 | 約 41 GiB | 収まる | 収まる |
| Qwen3-8B-Base | 8,190,735,360 | 約 79 GiB | ほぼ確実に不足 | 収まる見込み（torch の変更を伴う） |

内訳: bf16 の重み・勾配・AdamW 2 状態 = 8 × パラメータ数、logits (4, 1536, 151936) bf16 の 3 コピー分 約 5.2 GiB、root group の reduce-scatter 一時領域、checkpointing した各層の入力。
AdamW の状態は最初の `optimizer.step()` で作られるので、メモリ不足は 2 step 目以降に出る（試走は 8 step あるので検出できる）。実測は試走で行う。

## 6. 未確認（Colab の GPU で確認する）

- Colab の G4 ホストのドライバ版（CUDA 12.8 には R570 以上が必要）
- A100 80GB の割り当て（Pro+ でも保証されない。セル 1 が GPU 名と VRAM を表示・記録する）
- CUDA 版 torch を含む公式環境の作成時間とディスク使用量（推定 5〜6 GB）
- pip の cuDNN / NCCL が Colab のシステム版より優先されるか（セル 1 が cuDNN / NCCL の版を表示・記録する）
