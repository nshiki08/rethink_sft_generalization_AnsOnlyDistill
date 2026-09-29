"""Build the two notebooks from cells/*.md|*.py:
  notebooks/ansonly_distillation.ipynb  学習（cells/ の全ファイル、ファイル名順。評価セル 14 を除く）
  notebooks/ansonly_eval.ipynb          評価（EVAL_CELLS の順。学習ノートブックと同じセルのソースを使う）

Usage: python notebooks/ansonly_src/build_nb.py
"""
import ast, glob, os, sys
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.dirname(HERE)
EVAL_CELL = "14_section9_eval.py"
# 評価ノートブック: 認証・環境構築・設定・環境記録と HF ログイン・HF/変換の共通処理・評価。データ準備（3〜5）と学習（6-b〜8）は含めない
EVAL_CELLS = ["01_auth.py", "02_env_bootstrap.py", "03_config.py", "<NOTEBOOK_KIND=eval>", "04_section2_env_conditions.py",
              "06_helpers_runspec.py", "09_helpers_launch_monitor_hf.py", EVAL_CELL]
TRAIN_CELLS = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(HERE, "cells", "*"))) if os.path.basename(p) != EVAL_CELL]

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
    "09_helpers_launch_monitor_hf.py": "## 6. 短い学習・HF 保存・再開の試走\n\n6-a は起動・監視・HF 転送・再開の共通処理（セクション 7〜9、『再開』、評価ノートブックでも使う）。6-b が試走本体。",
    "11_section7_baseline.py": "## 7. baseline の確認または実行",
    "12_section8_search.py": "## 8. LR / epoch 探索（保留中）",
    "13_resume.py": "## 再開: Colab 切断後に HF の checkpoint から続ける\n\n切断後は **セル 0（認証）→ 1（環境構築）→ 設定セル → 2 → 3 → 4-a → 4-b → 6-a** を実行してから、`RESUME_RUN_ID` を設定してこのセルを実行する。",
    "15_section10_final.py": "## 9. 最終モデルの変換・HF 保存・読み込み確認・Model Card\n\n"
                             "評価ノートブック（`notebooks/ansonly_eval.ipynb`）で評価した結果が HF にあれば、Model Card に載せる。",
}
EVAL_SECTION_HEADERS = {
    "03_config.py": "## 設定\n\n評価で使う項目: `MODEL_KEY`（学習したモデル）、`PAPER_EVAL_AO_RUNS`（None なら HF にあるその MODEL_KEY の run すべて）、"
                    "`PAPER_EVAL_AO_STEPS`（\"paper\" = 論文の評価 step のうち HF にあるもの）。学習用の項目は評価では使わない。",
    "04_section2_env_conditions.py": "## 環境の記録・HF ログイン",
    "06_helpers_runspec.py": "## 共通処理（公式スクリプトの解析・HF）",
    "09_helpers_launch_monitor_hf.py": "## 共通処理（HF からの取得・変換）",
    EVAL_CELL: "## 評価（論文と同じ方法）\n\n公式の `evaluation/math_eval/math_eval_budget.py` を無変更で実行する"
               "（MATH500 avg@3, AIME24 avg@10, temperature 0.6, top_p 0.95, 最大 32768 token, math-verify）。"
               "HF の checkpoint を step ごとに取得・変換して評価し、結果は step ごとに HF の `runs/<run_id>/paper_eval/` に保存する。論文と同じく選択はしない。",
}
EVAL_INTRO = """# Answer-only (AO) 学生の評価ノートブック（Google Colab 用）

学習ノートブック（`notebooks/ansonly_distillation.ipynb`）が HF に保存した AO 学生の checkpoint を、論文と同じ方法で評価する。

- 評価: 公式 `evaluation/math_eval/math_eval_budget.py` を無変更で実行（MATH500 avg@3、AIME24 avg@10、temperature 0.6、top_p 0.95、最大 32768 token、math-verify）
- 比較: 公開 CoT 学生と Base は論文の値（App. D）を表に並べる。論文との差は tensor parallel 1（論文は 2 GPU）だけ
- GPU: A100（vLLM 0.8.5 は G4 に非対応）。学習とは別のセッションで実行してよい

## 実行順

セル 0（認証）→ 1（環境構築）→ 設定（`MODEL_KEY` などを学習と同じにする）→ 以降を順に実行。
データ準備と学習のセルは含まない。評価用の環境（Python 3.12 + vLLM 0.8.5）は評価セルが初回に作る。"""



def build(names, headers, out, id_prefix, intro=None):
    cells = [new_markdown_cell(intro)] if intro else []
    for name in names:
        if name == "<NOTEBOOK_KIND=eval>":
            cells.append(new_code_cell('%%ao\nNOTEBOOK_KIND = "eval"   # 評価ノートブック（セル 2 の勾配確認を行わない）'))
            continue
        src = open(os.path.join(HERE, "cells", name), encoding="utf-8").read().rstrip("\n")
        if name in headers:
            cells.append(new_markdown_cell(headers[name]))
        if name.endswith(".md"):
            cells.append(new_markdown_cell(src))
        elif name.endswith(".py"):
            ast.parse(src, filename=name)  # syntax check
            cells.append(new_code_cell(src if name in COLAB_KERNEL_CELLS else "%%ao\n" + src))
        else:
            raise ValueError(name)
    for i, c in enumerate(cells):
        c["id"] = f"{id_prefix}-{i:03d}"   # 再生成のたびに乱数 id が変わって差分が出ないよう固定する
    nb = new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU",
    })
    nbformat.validate(nb)
    with open(out, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("wrote", out, "cells:", len(cells))


if __name__ == "__main__":
    build(TRAIN_CELLS, SECTION_HEADERS, os.path.join(NB_DIR, "ansonly_distillation.ipynb"), "ao")
    build(EVAL_CELLS, {**{k: v for k, v in SECTION_HEADERS.items() if k in ("01_auth.py", "02_env_bootstrap.py")}, **EVAL_SECTION_HEADERS},
          os.path.join(NB_DIR, "ansonly_eval.ipynb"), "aoev", intro=EVAL_INTRO)
