# ハマりポイント

実際に詰まった点と、誤解しやすい点。環境（Python / GPU / 依存関係）は [environment.md](environment.md) にまとめる。

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

## 検証（CPU）

| 症状 | 原因 |
| --- | --- |
| torch 2.6 の FSDP2 が CPU で動かない（`'Stream' object has no attribute 'record_event'`） | CPU の Stream stub に不足。検証用 wrapper 側で穴埋めした（リポジトリは変更しない） |
| gloo で 8 プロセスを回すと勾配が 8 倍 | gloo は `reduce_scatter_tensor(op=AVG)` を SUM として扱う |
