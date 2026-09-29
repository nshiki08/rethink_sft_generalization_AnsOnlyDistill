# AO 蒸留ノートブックの構成

## 1. ファイル

| パス | 内容 |
| --- | --- |
| `notebooks/ansonly_distillation.ipynb` | Colab で開くノートブック。**生成物**。直接編集しない |
| `notebooks/ansonly_src/cells/NN_*.py / .md` | ノートブックのソース。ファイル名順に 1 セルずつ並ぶ |
| `notebooks/ansonly_src/build_nb.py` | ソースから `.ipynb` を生成する（構文チェック込み）。セクション見出しの markdown もここにある |
| `notebooks/ansonly_src/tests/run_cells_local.py` | 公式 pin の入った Python でセル 0 と設定〜5 を順に実行する（セル 1 は stub）。データと tokenizer を `$AO_TEST_WORK_DIR`（既定 `~/.cache/ao_nb_test`）に取得する |
| `notebooks/ansonly_src/tests/test_launch_monitor.py` | 偽 trainer（`tests/fake_trainer/`）で 起動・監視・中断・再開・manifest・アップロード方針 を検証する |
| `notebooks/ansonly_src/tests/test_eval_helpers.py` | セクション 9 の CPU で確認できる部分（公式評価スクリプトのデータパスとローダー、結果ファイルの読み取り、dev の採点、論文値） |
| `notebooks/ansonly_src/tests/test_env_bootstrap.py` | Colab と同じ構成（Python 3.13 のカーネル → uv の Python 3.12 公式環境 → 公式環境カーネル）でセル 0〜5 を CPU で実行する |
| `notebooks/README_ansonly.md` | 利用者向けの使い方 |
| `docs/ansonly/*.md` | 開発者向けの背景・決定事項・ハマりポイント |

編集の手順:

```bash
# 1. notebooks/ansonly_src/cells/ を編集
python notebooks/ansonly_src/build_nb.py                        # 2. ipynb を再生成
python notebooks/ansonly_src/tests/run_cells_local.py 08         # 3. CPU で設定〜セル 5（ドライラン）まで実行
python notebooks/ansonly_src/tests/test_launch_monitor.py        # 4. 起動・監視・再開の検証（3 の後に実行）
<python3.13 + jupyter_client + uv> notebooks/ansonly_src/tests/test_env_bootstrap.py 08   # 5. セル 1（環境構築）を含む Colab の流れ
```

3 と 4 は公式 pin の入った Python（CPU 版 torch で可）で実行する。5 は Colab のカーネルに相当する Python（3.13、`jupyter_client`, `ipython`, `uv`）で実行し、公式環境はセル 1 が作る。GPU・HF token は不要。

## 2. セルの役割

設定セル以降は `build_nb.py` が先頭に `%%ao` を付け、公式環境カーネルで実行される。

| セル | ファイル | 役割 | 主な出力（グローバル変数） |
| --- | --- | --- | --- |
| 0 | `01_auth.py` | **Colab カーネル**。Colab Secrets / 環境変数の `HF_TOKEN`, `WANDB_API_KEY` を環境変数に入れる（表示しない） | — |
| 1 | `02_env_bootstrap.py` | **Colab カーネル**。GPU 判定（プロファイル）、Fork の clone（`upstream` は push 無効）、uv で Python 3.12 の公式環境を作成・pin 導入・確認、公式環境カーネルを起動し `%%ao` を登録 | `AO_KERNEL`, `records/env_bootstrap.json` |
| 設定 | `03_config.py` | 利用者が編集する実験の設定（モデル・revision・実行モード・保存方針） | `SUPPORTED_MODELS` など |
| 2 | `04_section2_env_conditions.py` | 環境の記録、HF ログイン（whoami）、1 GPU の FSDP2 勾配確認、研究条件の表示 | `N_GPUS`, `PKG_VERSIONS`, `ENV_RECORD`, `HF_ACCOUNT_NAME`, `FSDP2_GRAD_CHECK` |
| 3 | `05_section3_data.py` | Math-CoT-20k 取得 → `</think>` 以降の最後の `\boxed{}` を抽出 → math-verify で参照正解と照合 → 全行監査 | `AO_PARQUET`, `AO_SHA256`, `AO_DATA_READY` |
| 4-a | `06_helpers_runspec.py` | 公式スクリプトの解析、run spec 生成、8 GPU 再現用の並べ替え | `build_run_spec()`, `prepare_train_file()` |
| 4-b | `07_section4_length_mask.py` | 公式 tokenizer・dataset クラスで全行の長さと loss mask を確認、`max_length` 決定、並べ替えの照合 | `MAX_LENGTH`, `MASK_CHECK_OK`, `VIEW_CHECK_OK` |
| 5 | `08_section5_dryrun.py` | 起動コマンド・step 数・保存・HF 保存先の確認（学習しない） | — |
| 6-a | `09_helpers_launch_monitor_hf.py` | 学習起動、監視スレッド、HF 転送・検証、再開、merge、実験記録 | `launch_training()` など |
| 6-b | `10_section6_trial.py` | 試走: 学習 → 保存 → 一時停止 → HF 転送 → 終了 → HF から取得 → 再開 → 参照 run と比較 | — |
| 7 / 8 | `11_*`, `12_*` | baseline / LR・epoch 探索 | — |
| 再開 | `13_resume.py` | HF の checkpoint から再開 | — |
| 9 | `14_section9_eval.py` | 9-a: 論文と同じ評価。評価用の別環境（Python 3.12 + vLLM 0.8.5）を作り、公式 `math_eval_budget.py` を無変更で実行（データの絶対パスはシンボリックリンク）。9-b: 任意の dev 評価 | `PAPER_EVAL_RESULTS` |
| 10 | `15_section10_final.py` | 最終モデル変換・HF 保存・Model Card | `FINAL_MODEL` |

## 3. 学習の起動の流れ

1. `build_run_spec()` が公式スクリプト（例 `training_scripts/Qwen3-1.7B_Math-CoT-20k_lr5e-5_ep8_bs256.sh`）を解析し、hydra 引数を公式と同じ値で組み立てる。公式と変える引数は `changes_vs_official` に理由付きで記録する
2. `launch_training()` が `python -m torch.distributed.run --nproc_per_node=N -m verl.trainer.fsdp_sft_trainer_ours <hydra 引数>` を新しい process group で起動する
3. `TrainingMonitor`（スレッド）が `global_step_<N>/` の保存完了を検知 → HF の `runs/<run_id>/global_step_<N>/` へ転送し sha256 manifest で検証 → 完了マーカー `ao_upload_verified.json` を置く。再開用の step（`upload_full_steps`）は optimizer 状態込み、それ以外は重みだけ（marker と manifest の `content`）。学習は転送中も進む。アップロード対象外の step はローカルから削除する。試走（`kill_after_step`）だけは SIGSTOP で学習を止めてから転送し、SIGTERM で終了して切断を模擬する
4. 再開時は HF から取得し manifest で検証してから `trainer.resume_mode=resume_path` で起動する

## 4. データの流れ

```
jasonrqh/Math-CoT-20k (revision 固定, 20,480 行)
  └─ セル 3: teacher_answer 列を追加 → data/Math-AO-20k.parquet（行順・message・advantage は元のまま）
       └─ セル 4-a: 行を並べ替え + advantage=N/8 → data/train_view/Math-AO-20k.emul-w8-nN.parquet（学習に渡す）
```

追加列: `teacher_answer`, `ao_extraction_status`, `ao_verify_status`, `ao_source_row`, `ao_n_boxes_after_think`。
