# AO 蒸留ノートブックの構成

## 1. ファイル

| パス | 内容 |
| --- | --- |
| `notebooks/ansonly_distillation.ipynb` | Colab で開くノートブック。**生成物**。直接編集しない |
| `notebooks/ansonly_src/cells/NN_*.py / .md` | ノートブックのソース。ファイル名順に 1 セルずつ並ぶ |
| `notebooks/ansonly_src/build_nb.py` | ソースから `.ipynb` を生成する（構文チェック込み）。セクション見出しの markdown もここにある |
| `notebooks/ansonly_src/tests/run_cells_local.py` | CPU でセル 1〜5 と helper を順に実行する（セル 2 は stub）。データと tokenizer を `$AO_TEST_WORK_DIR`（既定 `~/.cache/ao_nb_test`）に取得する |
| `notebooks/ansonly_src/tests/test_launch_monitor.py` | 偽 trainer（`tests/fake_trainer/`）で 起動・監視・中断・再開・manifest・アップロード方針 を検証する |
| `notebooks/README_ansonly.md` | 利用者向けの使い方 |
| `docs/ansonly/*.md` | 開発者向けの背景・決定事項・ハマりポイント |

編集の手順:

```bash
# 1. notebooks/ansonly_src/cells/ を編集
python notebooks/ansonly_src/build_nb.py                        # 2. ipynb を再生成
python notebooks/ansonly_src/tests/run_cells_local.py 08         # 3. CPU でセル 1〜5（ドライラン）まで実行
python notebooks/ansonly_src/tests/test_launch_monitor.py        # 4. 起動・監視・再開の検証（3 の後に実行）
```

テストに必要な Python パッケージは公式 `requirements.txt` と同じ（CPU 版 torch で可）。GPU・HF token は不要。

## 2. セルの役割

| セル | ファイル | 役割 | 主な出力（グローバル変数） |
| --- | --- | --- | --- |
| 0 | `01_auth.py` | Colab Secrets / 環境変数から `HF_TOKEN`, `WANDB_API_KEY` を読む。`huggingface_hub` を import 前に pin | `HF_LOGGED_IN`, `HF_ACCOUNT_NAME` |
| 設定 | `02_config.py` | 利用者が編集する全設定（モデル・revision・実行モード・保存方針・pin） | `SUPPORTED_MODELS`, `PIP_PINNED` など |
| 1 | `03_section1.py` | 研究条件の表示 | — |
| 2 | `04_section2_clone_env.py` | Fork の clone（`upstream` は push 無効）、GPU 情報、依存関係の導入、公式 trainer の import 確認、環境記録 | `REPO_DIR`, `N_GPUS`, `PKG_VERSIONS`, `ENV_RECORD` |
| 3 | `05_section3_data.py` | Math-CoT-20k 取得 → `</think>` 以降の最後の `\boxed{}` を抽出 → math-verify で参照正解と照合 → 全行監査 | `AO_PARQUET`, `AO_SHA256`, `AO_DATA_READY` |
| 4-a | `06_helpers_runspec.py` | 公式スクリプトの解析、run spec 生成、8 GPU 再現用の並べ替え | `build_run_spec()`, `prepare_train_file()` |
| 4-b | `07_section4_length_mask.py` | 公式 tokenizer・dataset クラスで全行の長さと loss mask を確認、`max_length` 決定、並べ替えの照合 | `MAX_LENGTH`, `MASK_CHECK_OK`, `VIEW_CHECK_OK` |
| 5 | `08_section5_dryrun.py` | 起動コマンド・step 数・保存・HF 保存先の確認（学習しない） | — |
| 6-a | `09_helpers_launch_monitor_hf.py` | 学習起動、監視スレッド、HF 転送・検証、再開、merge、実験記録 | `launch_training()` など |
| 6-b | `10_section6_trial.py` | 試走: 学習 → 保存 → 一時停止 → HF 転送 → 終了 → HF から取得 → 再開 → 参照 run と比較 | — |
| 7 / 8 | `11_*`, `12_*` | baseline / LR・epoch 探索 | — |
| 再開 | `13_resume.py` | HF の checkpoint から再開 | — |
| 9 / 10 | `14_*`, `15_*` | dev 評価、最終モデル変換・HF 保存・Model Card | — |

## 3. 学習の起動の流れ

1. `build_run_spec()` が公式スクリプト（例 `training_scripts/Qwen3-1.7B_Math-CoT-20k_lr5e-5_ep8_bs256.sh`）を解析し、hydra 引数を公式と同じ値で組み立てる。公式と変える引数は `changes_vs_official` に理由付きで記録する
2. `launch_training()` が `python -m torch.distributed.run --nproc_per_node=N -m verl.trainer.fsdp_sft_trainer_ours <hydra 引数>` を新しい process group で起動する
3. `TrainingMonitor`（スレッド）が `global_step_<N>/` の保存完了を検知 → HF の `runs/<run_id>/global_step_<N>/` へ転送し sha256 manifest で検証 → 完了マーカー `ao_upload_verified.json` を置く。学習は転送中も進む。アップロード対象外の step はローカルから削除する。試走（`kill_after_step`）だけは SIGSTOP で学習を止めてから転送し、SIGTERM で終了して切断を模擬する
4. 再開時は HF から取得し manifest で検証してから `trainer.resume_mode=resume_path` で起動する

## 4. データの流れ

```
jasonrqh/Math-CoT-20k (revision 固定, 20,480 行)
  └─ セル 3: teacher_answer 列を追加 → data/Math-AO-20k.parquet（行順・message・advantage は元のまま）
       └─ セル 4-a: 行を並べ替え + advantage=N/8 → data/train_view/Math-AO-20k.emul-w8-nN.parquet（学習に渡す）
```

追加列: `teacher_answer`, `ao_extraction_status`, `ao_verify_status`, `ao_source_row`, `ao_n_boxes_after_think`。
