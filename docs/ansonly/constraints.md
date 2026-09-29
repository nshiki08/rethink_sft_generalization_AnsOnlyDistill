# 守る制約（AO 蒸留ノートブック）

この Fork で Answer-only (AO) 蒸留を行うときに破ってはいけない条件。理由と確認方法を併記する。

## 1. 公式実装を変更しない

| 対象 | 規則 | 確認方法 |
| --- | --- | --- |
| `verl/`, `training_scripts/` | 1 行も変更しない。monkey patch もしない | `git diff 71a442e..HEAD -- verl training_scripts` が空。ノートブックのセル 2 も同じ確認をして `OFFICIAL_CODE_UNCHANGED` を記録する |
| 学習の起動 | 公式 trainer `verl.trainer.fsdp_sft_trainer_ours` を `python -m torch.distributed.run` で subprocess 起動し、公式スクリプト（`training_scripts/*.sh`）と同じ hydra 引数を渡す | セル 5 のドライランが公式スクリプトを実行時に解析し、公式との差分（`changes_vs_official`）を一覧表示する |
| 学習方式 | Full-parameter SFT のみ。LoRA / QLoRA は使わない | `model.lora_rank` を渡さない |
| 追加処理の置き場所 | ノートブック（ソースは `notebooks/ansonly_src/cells/`）だけ | 公式コードへ helper を足さない |

公式と変える hydra 引数は `changes_vs_official` に理由付きで記録される。現状の差分は次の通り:

- `data.response_key=teacher_answer`: AO target。これが唯一の実験条件の違い
- `data.max_length`: padding 長だけ。loss と勾配は変わらない
- `data.train_files` の並べ替え、`trainer.importance_sampling_mode=adv-only`: 8 GPU の再現用。[official_8gpu_emulation.md](official_8gpu_emulation.md) を参照
- 保存先と logger

## 2. Git

- push 先は Fork `nshiki08/rethink_sft_generalization_AnsOnlyDistill` のみ。オリジナル `Nebularaid2000/rethink_sft_generalization` への push は禁止。
- オリジナルは `upstream` として fetch 専用で登録し、push URL を `DISABLED_no_push_to_upstream` にする（セル 2 が自動で設定する）。push 前に `git remote -v` で確認する。
- データ本体（parquet）、モデル重み、checkpoint、認証情報は commit しない。
- ノートブック（`.ipynb`）は `notebooks/ansonly_src/build_nb.py` で生成する。`.ipynb` を直接編集しない（ソースと食い違う）。

## 3. 認証情報

- `HF_TOKEN`（write）と任意の `WANDB_API_KEY` は Colab Secrets か環境変数から、セル 0 で読む。値は表示しない（マスク表示のみ）。
- HF のアカウント名は `whoami` で決める。GitHub のユーザー名から推測しない。

## 4. Hugging Face 上の checkpoint

- 自動削除しない。過去 step を最新 step で上書きしない（`runs/<run_id>/global_step_<N>/` を step ごとに分け、完了マーカー `ao_upload_verified.json` がある step には再アップロードしない）。
- ローカルの checkpoint はアップロード対象外の step だけ削除してよい（Colab のディスク節約）。
- 容量削減のために HF 上のファイルを消すかどうかは利用者が決める。ノートブックは消さない。

## 5. 研究条件

- AO target は教師（Qwen3-32B）の `response` から抽出した最後の `\boxed{...}` を原文のまま使う。参照正解 `answer` 列で置き換えない。
- 入力 prompt（`message` 列）は公式と同一。末尾の「Please reason step by step...」も残す。
- 学習データは公開 `jasonrqh/Math-CoT-20k`（revision 固定）の 20,480 行をそのまま使う。行の削除・分割はしない。
- LR / epoch の候補は同じ Base から独立に学習する。選択は学習データでも最終テストでもない独立 dev で行う（dev は未確定。[open_items.md](open_items.md)）。
