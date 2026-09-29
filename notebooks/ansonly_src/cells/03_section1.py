# @title 1. 研究条件・対応モデルの確認
print("=== 対応モデル ===")
for _k, _v in SUPPORTED_MODELS.items():
    print(f"  {_k:16s} {_v['status']:9s} base={_v['base_repo']}@{_v['base_revision'][:8]}  cot={_v['cot_repo']}"
          f"{'/' + _v['cot_subfolder'] if _v['cot_subfolder'] else ''}@{_v['cot_revision'][:8]}")
    print(f"  {'':16s} note: {_v['note']}")
if MODEL_INFO["status"] != "supported":
    raise SystemExit(f"MODEL_KEY={MODEL_KEY} は blocked: {MODEL_INFO['note']}。設定を確認するまで進めない")

print("\n=== 比較対象 ===")
print(f"  Teacher            : {TEACHER['repo']} @ {TEACHER['revision'][:8]}（記録のみ）")
print(f"  Base Student       : {MODEL_INFO['base_repo']} @ {MODEL_INFO['base_revision'][:8]}")
print(f"  CoT Distill Student: {MODEL_INFO['cot_repo']}/{MODEL_INFO['cot_subfolder']} @ {MODEL_INFO['cot_revision'][:8]}（公開モデル。学習しない）")
print(f"  AO Distill Student : このノートブックで学習（Base から Full-parameter SFT）")

print("\n=== 公式基準設定（公開 CoT スクリプトを実行時に解析して表示する。ここは目安） ===")
print(f"  lr={OFFICIAL_BASELINE['lr']}, epochs={OFFICIAL_BASELINE['epochs']}  [{OFFICIAL_BASELINE['status']}]")
print("  探索候補（提案・未確定）:", SEARCH_GRID_PROPOSED)
print("  探索候補（確定）      :", SEARCH_GRID_CONFIRMED)
print("  このセッションで実行する候補:", SEARCH_RUN_LIST)

print("\n=== AO target とプロンプトの関係（実験記録に明記する） ===")
print("  入力 prompt (message 列) は公式と同一。末尾の『Please reason step by step, and put your final answer within \\boxed{}.』も変更しない。")
print("  target (teacher_answer 列) は教師 response の最後の \\boxed{...} を原文のまま用いる。CoT・説明文は含めない。")
print("  したがって AO 学生は『step by step』と指示されても推論を書かず \\boxed{答え} + EOS を出力するよう学習される。")
print("  公式の Math-NoCoT-20k（<think> を除き最終まとめ＋回答を残す）とは異なる。参照正解 answer 列は照合にだけ使い、target に代用しない。")

print(f"\n=== 公式 CoT 学生の学習条件（{OFFICIAL_WORLD_SIZE} GPU）への合わせ込み: {'ON' if EMULATE_OFFICIAL_WORLD_SIZE else 'OFF'} ===")
print(f"  公式 CoT 学生は {OFFICIAL_WORLD_SIZE} GPU で学習された（README.md:101）。公式 trainer は 1 GPU 内の micro batch の勾配を和で蓄積し、GPU 間は平均する。")
print(f"  そのまま GPU 台数 N で動かすと、micro batch（4 行）の組み合わせと勾配の大きさ（{OFFICIAL_WORLD_SIZE}/N 倍）が公式と変わる。")
if EMULATE_OFFICIAL_WORLD_SIZE:
    print("  ON のとき: 学習用 parquet の行を並べ替えて各 micro batch を公式と同じ 4 行にし、adv-only + advantage=N/8 で loss を N/8 倍して勾配を公式と同じ大きさにする。")
    print("  学習手順の違いは target 列（teacher_answer）だけになる。残る差は丸め誤差程度（GPU の種類、bf16 の加算順序など。ドライランで一覧表示）。")
