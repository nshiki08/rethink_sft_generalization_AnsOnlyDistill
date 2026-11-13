# ハマりポイント

実際に詰まった点と、誤解しやすい点。環境の構成と根拠は [environment.md](environment.md)。

## 環境

| 誤解・症状 | 事実 | 対処 |
| --- | --- | --- |
| Colab のカーネルに公式の pin を pip で入れる | Colab は Python 3.13（2026-08 から）。numpy 1.26.4 / ray 2.43.0 は cp312 まで | セル 1 が uv で Python 3.12 の公式環境を作り、2 つ目のカーネルで実行する |
| 学習の subprocess だけ公式環境で動かせば足りる | セル内でも公式コード（dataset クラス、math-verify、tokenizer）を使う。`import verl` は ray を必要とする | 設定セル以降を公式環境カーネルで実行（`%%ao`） |
| G4 でも公式 torch 2.6.0 が動く | import はできるが sm_120 のコードが無く、CUDA カーネルの実行で失敗する（"no kernel image is available"） | A100 なら公式 pin のまま。G4 は `ALLOW_BLACKWELL_TORCH_DEVIATION=True`（torch 2.7.1+cu128）で差分として記録 |
| セル 1 が「torch に compute capability … 用のコードが無い: []」で止まる（以前の表示） | torch が CUDA を初期化できないと arch 一覧が空になり、原因（ドライバと torch の CUDA 版の不一致、GPU の切断、CPU 版の torch）と違う文が出ていた | `gpu_probe_problem()` が 2 つの場合を分け、初期化できないときは torch のエラー文とドライバの版を表示する |
| flash-attn の wheel を「torch の版」だけで選ぶ | 2.7.4.post1 の torch2.6 用 wheel は sm_80/90 のみ、torch2.7 用 wheel は sm_120 まで含む。ABI も違う（FALSE / TRUE） | プロファイルごとに wheel URL を固定 |
| 1 GPU の FSDP2 は reduce が恒等なので安全 | world size 1 の NCCL AVG で勾配が 0 になる未解決報告がある | セクション 2 で FSDP 無しとの勾配比較。OK でないと学習しない |
| Colab の uv 設定がそのまま使われる | Colab は `UV_*` で制約ファイルを指定していることがあり、Colab 既定の版に固定される | 公式環境の作成では `UV_*` を外し `--no-config` |
| Unix socket のパスに作業ディレクトリを使う | パス長の上限（108 byte）を超えると ZMQ が失敗する | カーネル通信は `/tmp/aok<pid>` |
| pandas は公式 pin に従う | `requirements.txt` に pandas が無い。放置すると導入時期で 3.x になる | `pandas==2.3.3` に固定（notebook の判断として記録） |
| 生成した AO parquet の sha256 が環境によって変わる | parquet のメタデータ `created_by` に pyarrow の版が入る（内容は同一。pyarrow 25.0.1 と 21.0.0 で確認） | 再開時の sha256 照合は同じ公式環境（pyarrow==21.0.0）で行う。内容の同一性は `Table.equals` で確認できる |
| セル 1 の再実行で GPU メモリが残る | 前回の公式環境カーネルと、別の process group で動く学習プロセス（torchrun）が残る | 両方の process group ID を pid ファイルに記録し、cmdline を確認してから終了する |
| G4 用に torch だけ上げる | torch 2.7.1 は `sympy>=1.13.3` を要求し、公式の `sympy==1.13.1` と依存解決できない | blackwell プロファイルは sympy 1.13.3（math-verify の結果が変わらないことを全行で確認済み）。プロファイルごとの依存解決は `test_env_bootstrap.py` が確認する |
| L4 で「torch に sm_89 が無い」 | torch 2.6.0 は sm_86 までのコードを持ち、同じ major の sm_89 で動く | アーキテクチャの判定は「同じ major で minor が実機以下」か PTX で行う |

## 学習条件

| 誤解・症状 | 事実 | 対処 |
| --- | --- | --- |
| 「勾配累積で 1 GPU でも 8 GPU と同じ」 | 公式 trainer は micro batch の勾配を和で累積し、GPU 間は平均する。1 GPU では勾配が 8 倍、micro batch の組み合わせも変わる | [official_8gpu_emulation.md](official_8gpu_emulation.md) の並べ替え + adv-only |
| `optim.clip_grad=8` で勾配の大きさを補正できる | AdamW の eps（1e-8、変更不可）の効き方が変わり 8% 程度ずれる | 採用しない |
| `trainer.seed` で乱数が揃う | trainer は seed を使っていない（grep で参照なし）。dropout も無い。`shuffle=False` なので全 epoch 同じ順 | 不要 |
| 公式 `max_length=20000` のまま 1 GPU に載らない | dataset が全行を `max_length` まで padding する。AO の最長行は 1488 token | `MAX_LENGTH_MODE="auto_fit"`（1536）。pad 位置は loss_mask=0 で flash-attn から除かれるので loss・勾配は変わらない |
| micro batch を小さくしてメモリを減らす | token 平均の単位が変わり公式と別の学習になる | GPU を大きくする。変える場合は `EMULATE_OFFICIAL_WORLD_SIZE=False` と差分記録 |
| `train/loss` が公式の 1/8 | adv-only で loss を N/8 倍しているため | `train/loss_official_scale`（8/N 倍した値）を見る |
| `train/grad_norm` が公式曲線と合わない | 公式 8 GPU のログは各 GPU の shard ノルムの平均。1 GPU は全体ノルム | 比較に使わない |
| 別 GPU 台数の checkpoint から再開 | ファイル名・shard が world_size ごと | 同じ N で再開する（`check_resume_compatibility` が止める） |
| 重みだけの checkpoint から再開 | optimizer 状態が無い（trainer は `load_contents` に optimizer を含めて読む） | 再開候補は optimizer 状態込みの step だけ。重みだけの step は変換・分析用 |
| メモリ不足が 1 step 目で出ない | AdamW の状態は最初の `optimizer.step()` で作られる | 試走は 2 step 以上（既定 8 step） |

## データ

| 誤解・症状 | 事実 | 対処 |
| --- | --- | --- |
| `response` 全体の最後の `\boxed{}` を取る | `</think>` の前（思考中）の `\boxed{}` を拾う危険がある | `</think>` 以降の最後の `\boxed{}`（公式 `last_boxed_only_string`）|
| 参照正解 `answer` と一致しない行を除く・置き換える | 行を消すと公式と学習データが変わる。置換は教師の出力ではなくなる | 教師の原文を残し、不一致行は `AO_ACKNOWLEDGED_VERIFY_MISMATCH_ROWS` で明示承認する |
| math-verify の不一致 = 教師の誤り | 方程式の答え（楕円・円の式など）は、同値でも math-verify の比較が失敗する | 行 12364, 16249, 16611 は同値な式（[open_items.md](open_items.md)） |
| 公式の Math-NoCoT-20k を AO として使う | NoCoT は `<think>` を除いた「最終まとめ + 回答」で、答えだけではない | AO は `\boxed{...}` のみを target にする |

## ノートブックの実装

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| 試走で「保存完了 → HF 転送 → 終了」の前に学習が先に進む | 転送中も学習が進む | 試走では SIGSTOP で停止してから転送、SIGTERM で終了 |
| 再開元 step の再アップロードが衝突エラーになる | HF に完了済みの step | 完了マーカーを見て「転送済み」とする。上書きしない |
| 保存途中の checkpoint を転送してしまう | trainer はファイルを順に書く | `checkpoint_is_complete()`: 全ファイルの存在・zip の整合・サイズが 2 回の走査で不変 |
| カーネルの `huggingface_hub` と学習側の版が違う | Colab は起動時に別版を import 済み | セル 0 で import 前に pin。違う場合は警告 |
| `dict()` got multiple values for keyword argument | 監査 jsonl で `row` を二重指定 | 辞書を別に組み立てる |
| 設定セルを実行し直すと、試走・本学習が「`HF_CKPT_REPO_ID` が未設定」で止まり、再開は「checkpoint が無い」で止まる（以前の動作） | 4-a が `HF_CKPT_REPO_ID` を書き換えていたので、設定セルの再実行で `None` に戻った | `resolve_ckpt_repo_id()` が使う直前に決める。HF を使うコードは `HF_CKPT_REPO_ID` を直接読まない。再開は「未ログイン」「repo を参照できない」「ファイル一覧を取得できない」「完了マーカーを取得できない」「run が無い」「再開できる step が無い」を別の文で止める |
| 切断すると loss の推移が残らない（以前の動作） | loss はメモリ上にあり、ファイルに書くのは学習終了後だった。ログの書き込みもバッファされていた | 1 step ごとに `.metrics.jsonl` へ追記、ログも 1 行ごとに書き出す。写し先（Drive）へは `LogMirror` が別スレッドで写す |
| Drive のパスに書いたのに Drive に無い | Drive をマウントしていない状態で `/content/drive/...` にフォルダを作ると、ランタイムのディスクに普通のフォルダができる（切断で消える） | `mirror_dir_problem()`: `/content/drive` 配下でマウントされていなければ書かずに警告する |
| 写し先を相対パスで書く | セル 0（Colab のカーネル、`/content`）と公式環境のカーネル（作業フォルダ）で別の場所を指し、マウントの検査をすり抜けてランタイムのディスクに書く | セル 0 は相対パスの `AO_DRIVE_LOG_DIR` に警告を出し、直さずに渡す。`mirror_dir_problem()` は絶対パスでない写し先に書かない。設定セルは `LOG_MIRROR_DIR` が絶対パスであることを確かめる。Colab で Drive の外の絶対パスを登録したときは、セル 0 が「ランタイムのディスクにある」と警告する |
| セル 1 の後に `AO_DRIVE_LOG_DIR` を登録してセル 0 を実行し直しても、学習ログが写されない | 公式環境のカーネルの環境変数は、セル 1 が起動した時点の写し | セル 0 が、動いている公式環境のカーネルにも値を渡す |
| 写し先のファイルを毎回まるごと上書きする | 上書きの途中で切断すると、それまでに写した内容まで失う | ログと loss の推移は `append_missing_tail()` が写し先に無い末尾だけを追記し、既にある内容は消さない。実験記録（`mirror_record()`）は一時名に書いてから置き換える |
| HF と通信できなかっただけなのに、完了済みの step を「HF に無い」「転送が不完全」と扱う | 一覧や完了マーカーの取得の失敗を、空の一覧・未完了と同じに扱っていた。再開が古い step を選ぶ、同じ step を転送し直して上書きする、という動作になり得る | 再開（`resume_run_from_hf`）、転送（`upload_checkpoint`）、本学習の直前確認（`preflight_train`）は、一覧を取得できなければ進まない。再開は完了マーカーを取得できないときも止まる。表示だけの箇所（ドライラン、探索の一覧）は従来どおり空として扱う |
| wandb（offline）の記録がどの run のものか分からない | 全 run が同じ `WANDB_DIR` に書いていた | `WANDB_DIR` を run ごとの `wandb/<run_id>/` にする（再開時もその run のフォルダ） |

## 検証（CPU）

| 症状 | 原因 |
| --- | --- |
| torch 2.6 の FSDP2 が CPU で動かない（`'Stream' object has no attribute 'record_event'`） | CPU の Stream stub に不足。検証用 wrapper 側で穴埋めした（リポジトリは変更しない） |
| gloo で 8 プロセスを回すと勾配が 8 倍 | gloo は `reduce_scatter_tensor(op=AVG)` を SUM として扱う |
