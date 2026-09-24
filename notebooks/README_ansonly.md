# Answer-only (AO) 蒸留ノートブックの使い方

対象: `notebooks/ansonly_distillation.ipynb`（Google Colab 用）。
公式実装 `verl.trainer.fsdp_sft_trainer_ours` と `training_scripts/` は変更せず、ノートブックから既存の引数を渡して AO 学生を学習する。

## 1. Colab で開く

1. GitHub の Fork `nshiki08/rethink_sft_generalization_AnsOnlyDistill` で `notebooks/ansonly_distillation.ipynb` を開き、「Open in Colab」または
   `https://colab.research.google.com/github/nshiki08/rethink_sft_generalization_AnsOnlyDistill/blob/<branch>/notebooks/ansonly_distillation.ipynb` を開く。
2. ランタイム → GPU（A100 / L4 など Ampere 以上。T4 は flash-attn 2 が動かないので不可）。
3. Colab の「シークレット」に以下を登録し、ノートブックからのアクセスを許可する。
   - `HF_TOKEN`（write 権限。checkpoint と最終モデルの保存に使う）
   - `WANDB_API_KEY`（任意。無ければ公式と同じ offline mode）

## 2. 設定

すべて **設定セル** で行う。主な項目:

| 項目 | 変数 | 既定 |
| --- | --- | --- |
| 実行モード | `RUN_MODE` | `"dry_run"`（学習も HF 書き込みもしない） / `"trial"` / `"train"` |
| run の種類 | `RUN_KIND` | `"baseline"`（公式 CoT 設定 lr 5e-5 / 8 epoch）/ `"search"` |
| 対象モデル | `MODEL_KEY` | `"Qwen3-1.7B"`（`SUPPORTED_MODELS` から選ぶ。Qwen2.5 系は理由付きで blocked） |
| 探索候補 | `SEARCH_GRID_PROPOSED`（提案）/ `SEARCH_GRID_CONFIRMED`（確定）/ `SEARCH_RUN_LIST`（実行する組） | 提案のみ |
| 最大系列長 | `MAX_LENGTH_MODE` | `"auto_fit"`（全行が収まる長さ。`"official"` で 20000 に戻せる） |
| 保存周期 | `SAVE_FREQ` | 40（正整数。公式 CoT は 10） |
| HF 保存先 | `HF_CKPT_REPO_ID` / `HF_FINAL_MODEL_REPO_ID` | `None` → HF の whoami から決める |
| 試走 | `TRIAL_NUM_ROWS`, `TRIAL_EPOCHS`, `TRIAL_SAVE_FREQ`, `TRIAL_KILL_AFTER_STEP` | 512 行 / 3 epoch / 3 / 3 |
| 再開 | `RESUME_RUN_ID`, `RESUME_STEP` | `None` / `"latest"` |
| dev 評価 | `DEV_EVAL_ENABLED`, `DEV_EVAL_SOURCE` など | 無効（dev 未確定） |

## 3. 実行順（初回）

セル 0（認証）→ 設定 → 1 → 2 → 3 → 4-a → 4-b → 5（ドライラン）。ここまでは GPU 不要で、HF への書き込みもしない。

- 3 で `AO_DATA_READY`、4-b で `MASK_CHECK_OK` が `True` にならなければ、本学習のセルは開始しない。
- 5 で起動コマンド・想定 step 数・保存回数・HF 保存先・衝突の有無を確認する。

次に `RUN_MODE="trial"` にして 6-a → 6-b（試走: 学習 → HF 転送 → プロセス終了 → HF から取得 → 新プロセスで再開）。
試走の run は `trial-` で始まり、本学習と別ディレクトリ・別データ（先頭 512 行）を使う。試走の checkpoint は本学習に使わない。

本学習は `RUN_MODE="train"`、`RUN_KIND="baseline"` で 7、`RUN_KIND="search"` かつ `SEARCH_RUN_LIST` を設定して 8。
学習中は 1 セルが終了まで動き続け、監視スレッドが保存完了ごとに checkpoint を HF の
`runs/<run_id>/global_step_<N>/` へ転送・検証し、完了マーカー `ao_upload_verified.json` を置く。過去 step は上書き・削除しない。

## 4. 切断後の再開

1. 新しいランタイムでセル 0 → 設定 → 1 → 2 → 3 → 4-a → 4-b → 6-a を実行する（同じ Fork commit・同じデータ・同じ Base revision が復元される）。
2. 設定セルで `RUN_MODE="train"`、`RESUME_RUN_ID="<run_id>"`（`RESUME_STEP` は `"latest"` か整数）を設定する。
3. 「再開」セルを実行する。HF の完了済み checkpoint 一覧を表示し、選択した step を取得して manifest（サイズ・sha256）で検証し、
   GPU 台数（FSDP world_size）・データ sha256・Base revision の互換性を確認してから
   `trainer.resume_mode=resume_path trainer.resume_from_path=...` で再開する。総 epoch 数・総 step 数は保存時の値を維持する。
   最後の保存以降の step は再実行になる。

## 5. 成果物

- ローカル: `/content/ao_work/{data,models,ckpt,log,records,merged}`（Colab のディスク。切断で消える）
- HF checkpoint repo: `runs/<run_id>/global_step_<N>/`（model/optimizer/extra の shard, `data.pt`, `huggingface/`, `ao_ckpt_manifest.json`, `ao_upload_verified.json`）と `runs/<run_id>/experiment_record_*.json`
- HF 最終モデル repo: `step<N>/`（`verl.model_merger` で変換した HF 形式）と `README.md`（Model Card）

## 6. 未検証の項目

このノートブックは GPU の無い環境で静的に検証した（構文、データ抽出・監査の全行実行、tokenizer での長さ計算、公式 dataset クラスによる loss mask 検証、公式 trainer の import）。
以下は Colab で各セルを実行し、ログを確認するまで「動作確認済み」ではない。

- 依存関係のインストールと flash-attn wheel の適用（Colab の Python / torch / CUDA の組み合わせに依存）
- 試走（学習・保存・HF 転送・終了・再開）、本学習、GPU メモリ、所要時間
- `verl.model_merger` による変換と最終モデルの読み込み・生成
