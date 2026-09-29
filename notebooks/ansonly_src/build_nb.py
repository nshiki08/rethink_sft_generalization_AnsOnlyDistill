"""Build notebooks/ansonly_distillation.ipynb from cells/*.md|*.py (order = filename sort).

Usage: python notebooks/ansonly_src/build_nb.py [out.ipynb]   (default: notebooks/ansonly_distillation.ipynb)
"""
import ast, glob, os, sys
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "ansonly_distillation.ipynb")

SECTION_HEADERS = {
    "03_section1.py": "## 1. 設定・研究条件・対応モデルの確認",
    "04_section2_clone_env.py": "## 2. Fork の clone・revision 固定・環境構築",
    "05_section3_data.py": "## 3. データ取得・AO 抽出・全行監査\n\n`teacher_answer` 列は `response` の `</think>` 以降の最後の `\\boxed{...}` を原文のまま用いる。抽出失敗行は `null` のまま残し（削除も `answer` 代用もしない）、ゲート `AO_DATA_READY` で本学習を止める。",
    "06_helpers_runspec.py": "## 4. 長さ・target・loss mask・実効設定の確認",
    "08_section5_dryrun.py": "## 5. ドライラン（既定の実行モード）",
    "09_helpers_launch_monitor_hf.py": "## 6. 短い学習・HF 保存・再開の試走\n\n6-a は起動・監視・HF 転送・再開の共通処理（セクション 7〜10 と『再開』でも使う）。6-b が試走本体。",
    "11_section7_baseline.py": "## 7. baseline の確認または実行",
    "12_section8_search.py": "## 8. LR / epoch 探索",
    "13_resume.py": "## 再開: Colab 切断後に HF の checkpoint から続ける\n\n切断後は **セル 0（認証）→ 設定セル → 1 → 2 → 3 → 4-a → 4-b → 6-a** を実行してから、`RESUME_RUN_ID` を設定してこのセルを実行する。",
    "14_section9_dev_eval.py": "## 9. dev 評価と結果集計",
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
        cells.append(new_code_cell(src))
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
