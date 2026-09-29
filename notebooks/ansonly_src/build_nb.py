"""Build notebooks/ansonly_distillation.ipynb from cells/*.md|*.py (order = filename sort).

Usage: python notebooks/ansonly_src/build_nb.py [out.ipynb]   (default: notebooks/ansonly_distillation.ipynb)
"""
import ast, glob, os, sys
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "ansonly_distillation.ipynb")

# Colab のカーネル（Python 3.13）で実行するセル。それ以外の .py セルには先頭に %%ao を付け、セル 1 が起動する公式環境（Python 3.12 + 公式 pin）のカーネルで実行する
COLAB_KERNEL_CELLS = {"01_auth.py", "02_env_bootstrap.py"}

SECTION_HEADERS = {
    "01_auth.py": "## 0. 認証\n\nこのセルとセル 1 は Colab のカーネルで実行する。",
    "02_env_bootstrap.py": "## 1. 環境構築（Fork の clone・公式環境の作成・公式環境カーネルの起動）\n\n"
                           "Colab の Python（3.13）には公式の pin が入らないので、Python 3.12 の別環境に公式の pin を入れ、その環境で 2 つ目のカーネルを起動する。"
                           "**設定セル以降の `%%ao` で始まるセルは公式環境のカーネルで実行される。** Colab の『セッションを再起動』をした場合はセル 0 から実行し直す。",
    "03_config.py": "## 設定",
    "04_section2_env_conditions.py": "## 2. 環境の記録・HF ログイン・1 GPU の FSDP2 勾配確認・研究条件",
    "05_section3_data.py": "## 3. データ取得・AO 抽出・全行監査\n\n`teacher_answer` 列は `response` の `</think>` 以降の最後の `\\boxed{...}` を原文のまま用いる。抽出失敗行は `null` のまま残し（削除も `answer` 代用もしない）、ゲート `AO_DATA_READY` で本学習を止める。",
    "06_helpers_runspec.py": "## 4. 長さ・target・loss mask・実効設定の確認",
    "08_section5_dryrun.py": "## 5. ドライラン（既定の実行モード）",
    "09_helpers_launch_monitor_hf.py": "## 6. 短い学習・HF 保存・再開の試走\n\n6-a は起動・監視・HF 転送・再開の共通処理（セクション 7〜10 と『再開』でも使う）。6-b が試走本体。",
    "11_section7_baseline.py": "## 7. baseline の確認または実行",
    "12_section8_search.py": "## 8. LR / epoch 探索（保留中）",
    "13_resume.py": "## 再開: Colab 切断後に HF の checkpoint から続ける\n\n切断後は **セル 0（認証）→ 1（環境構築）→ 設定セル → 2 → 3 → 4-a → 4-b → 6-a** を実行してから、`RESUME_RUN_ID` を設定してこのセルを実行する。",
    "14_section9_eval.py": "## 9. 評価（論文と同じ方法）\n\n公式の `evaluation/math_eval/math_eval_budget.py` を無変更で実行する（MATH500 avg@3, AIME24 avg@10, temperature 0.6, 最大 32768 token）。"
                           "論文と同じく各 step の推移を記録し、条件や checkpoint の選択はしない。",
    "15_section10_final.py": "## 10. 最終モデルの変換・HF 保存・読み込み確認・Model Card",
}

cells = []
for path in sorted(glob.glob(os.path.join(HERE, "cells", "*"))):
    name = os.path.basename(path)
    src = open(path, encoding="utf-8").read().rstrip("\n")
    if name in SECTION_HEADERS:
        cells.append(new_markdown_cell(SECTION_HEADERS[name]))
    if name.endswith(".md"):
        cells.append(new_markdown_cell(src))
    elif name.endswith(".py"):
        ast.parse(src, filename=name)  # syntax check
        cells.append(new_code_cell(src if name in COLAB_KERNEL_CELLS else "%%ao\n" + src))
    else:
        raise ValueError(name)

for _i, _c in enumerate(cells):
    _c["id"] = f"ao-{_i:03d}"   # 再生成のたびに乱数 id が変わって差分が出ないよう固定する
nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": [], "toc_visible": True},
    "accelerator": "GPU",
})
nbformat.validate(nb)
with open(OUT, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells:", len(cells))
