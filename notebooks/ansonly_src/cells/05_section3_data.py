# @title 3. データ取得・AO 抽出・全行監査（元 20,480 行・行順・message・advantage を維持し teacher_answer 列を追加）
import hashlib, json, os, re, collections, time
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
import pandas as pd
import multiprocessing
from huggingface_hub import hf_hub_download

# 既存の boxed 抽出・照合処理を流用する（verl/utils/reward_score/math_verify_ours.py: last_boxed_only_string + math-verify）
from verl.utils.reward_score.math_verify_ours import last_boxed_only_string, compute_score as mv_compute_score



# ---- 3.1 取得（revision 固定、sha256 照合） ----
RAW_PARQUET = hf_hub_download(repo_id=DATASET_REPO, filename=DATASET_FILE, repo_type="dataset", revision=DATASET_REVISION,
                              local_dir=f"{DATA_DIR}/raw")
RAW_SHA256 = sha256_of(RAW_PARQUET)
print("raw parquet:", RAW_PARQUET, "\nsha256:", RAW_SHA256)
assert RAW_SHA256 == DATASET_EXPECTED_SHA256, "Math-CoT-20k.parquet の sha256 が想定と異なる。revision を確認する"

RAW_TABLE = pq.read_table(RAW_PARQUET)
assert RAW_TABLE.num_rows == DATASET_EXPECTED_ROWS, RAW_TABLE.num_rows
for _c in ("message", "response", "answer", "advantage"):
    assert _c in RAW_TABLE.column_names, _c
print("columns:", RAW_TABLE.column_names, "| rows:", RAW_TABLE.num_rows)
_raw_df = RAW_TABLE.select(["response", "answer"]).to_pandas()


# ---- 3.2 抽出（監査用に全 box を列挙し、target は既存 helper の last_boxed_only_string で取る） ----
def find_all_boxed(text):
    """監査用: \\boxed の出現を先頭から列挙する。(位置, 文字列 | 'NOBRACE' | None=閉じ括弧なし)"""
    out, start = [], 0
    while True:
        idx = text.find("\\boxed", start)
        if idx < 0:
            break
        k = idx + len("\\boxed")
        while k < len(text) and text[k] == " ":
            k += 1
        if k < len(text) and text[k] == "{":
            depth, i, right = 0, k, None
            while i < len(text):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        right = i
                        break
                i += 1
            if right is None:
                out.append((idx, None)); start = k + 1
            else:
                out.append((idx, text[idx:right + 1])); start = right + 1
        else:
            out.append((idx, "NOBRACE")); start = k
    return out


def extract_ao_target(response, style):
    """response（教師 CoT+回答）から AO target を抽出する。戻り値: (target or None, status, info)"""
    info = dict(n_think_open=response.count("<think>"), n_think_close=response.count("</think>"))
    after = response.split("</think>")[-1] if info["n_think_close"] >= 1 else response
    boxes_after = find_all_boxed(after)
    info["n_boxes_after_think"] = len(boxes_after)
    info["n_boxes_total"] = len(find_all_boxed(response))
    info["nobrace_form_after_think"] = any(b[1] == "NOBRACE" for b in boxes_after)
    info["unclosed_box_after_think"] = any(b[1] is None for b in boxes_after)
    distinct = [b[1] for b in boxes_after if isinstance(b[1], str) and b[1] != "NOBRACE"]
    info["n_distinct_boxes_after_think"] = len(set(distinct))
    info["first_box_after_think"] = distinct[0] if distinct else None
    last_box = last_boxed_only_string(after)             # 既存 helper（入れ子括弧対応）
    info["last_box_equals_response_last_box"] = (last_box == last_boxed_only_string(response))
    if last_box is None:
        return None, "no_box", info
    inner = last_box[len("\\boxed{"):-1]
    if inner.strip() == "":
        return None, "empty_box", info
    if style == "last_boxed_verbatim":
        target = last_box
    elif style == "boxed_content":
        target = inner
    elif style == "last_boxed_line":
        pos = after.rfind(last_box)
        ls = after.rfind("\n", 0, pos) + 1
        le = after.find("\n", pos + len(last_box))
        target = after[ls: (le if le >= 0 else len(after))].strip()
    else:
        raise ValueError(style)
    return target, "ok", info


def _verify_row(args):
    target, gold = args
    if target is None:
        return "not_extracted"
    try:
        s = mv_compute_score(solution_str=target, ground_truth=gold)   # 既存の照合処理（gold を $..$ で包み math-verify）
    except Exception as e:  # noqa
        return "verify_error:" + type(e).__name__
    return "match" if float(s) == 1.0 else "mismatch"


t0 = time.time()
_targets, _status, _infos = [], [], []
for resp in _raw_df["response"].tolist():
    t, s, i = extract_ao_target(resp, AO_TARGET_STYLE)
    _targets.append(t); _status.append(s); _infos.append(i)
print(f"extraction done in {time.time() - t0:.1f}s")

t0 = time.time()
# セル内で定義した関数を worker で使うため fork を明示（Colab/Linux）。spawn 環境では動かない
with multiprocessing.get_context("fork").Pool(max(1, min(8, os.cpu_count() or 1))) as _pool:
    _verify = _pool.map(_verify_row, list(zip(_targets, _raw_df["answer"].tolist())), chunksize=64)
print(f"math-verify done in {time.time() - t0:.1f}s")

AUDIT = pd.DataFrame(_infos)
AUDIT.insert(0, "row", range(len(AUDIT)))
AUDIT["extraction_status"] = _status
AUDIT["teacher_answer"] = _targets
AUDIT["answer"] = _raw_df["answer"].values
AUDIT["verify_status"] = _verify
AUDIT["teacher_answer_chars"] = AUDIT["teacher_answer"].map(lambda x: len(x) if x else 0)

# ---- 3.3 監査集計とゲート ----
_summary = dict(
    rows=int(len(AUDIT)), target_style=AO_TARGET_STYLE, dataset_repo=DATASET_REPO, dataset_revision=DATASET_REVISION, raw_sha256=RAW_SHA256,
    extraction_status=AUDIT["extraction_status"].value_counts().to_dict(),
    verify_status=AUDIT["verify_status"].value_counts().to_dict(),
    think_tag_counts=dict(open=AUDIT["n_think_open"].value_counts().to_dict(), close=AUDIT["n_think_close"].value_counts().to_dict()),
    n_boxes_after_think=AUDIT["n_boxes_after_think"].value_counts().sort_index().to_dict(),
    n_distinct_boxes_after_think=AUDIT["n_distinct_boxes_after_think"].value_counts().sort_index().to_dict(),
    rows_multi_distinct_boxes=int((AUDIT["n_distinct_boxes_after_think"] > 1).sum()),
    rows_nobrace_form_after_think=int(AUDIT["nobrace_form_after_think"].sum()),
    rows_unclosed_box_after_think=int(AUDIT["unclosed_box_after_think"].sum()),
    rows_last_box_differs_from_response_last_box=int((~AUDIT["last_box_equals_response_last_box"]).sum()),
    teacher_answer_chars=dict(min=int(AUDIT["teacher_answer_chars"].min()), mean=float(AUDIT["teacher_answer_chars"].mean()), max=int(AUDIT["teacher_answer_chars"].max())),
)
_failed = AUDIT[AUDIT["extraction_status"] != "ok"]
_mismatch = AUDIT[AUDIT["verify_status"] != "match"]
_unacknowledged = sorted(set(_mismatch["row"].tolist()) - set(AO_ACKNOWLEDGED_VERIFY_MISMATCH_ROWS))
_stale_ack = sorted(set(AO_ACKNOWLEDGED_VERIFY_MISMATCH_ROWS) - set(_mismatch["row"].tolist()))
_summary.update(rows_extraction_failed=_failed["row"].tolist(), rows_verify_mismatch=_mismatch["row"].tolist(),
                rows_verify_mismatch_unacknowledged=_unacknowledged, acknowledged_rows_not_mismatching=_stale_ack)
print(json.dumps(_summary, indent=1, ensure_ascii=False))

pd.set_option("display.max_colwidth", 120)
if len(_mismatch):
    print("\n--- math-verify 不一致行（教師原文の box と参照正解。代用も削除もしない） ---")
    print(_mismatch[["row", "teacher_answer", "answer", "verify_status", "n_distinct_boxes_after_think"]].to_string(index=False))
if len(_failed):
    print("\n--- 抽出失敗行 ---")
    print(_failed[["row", "extraction_status", "n_boxes_after_think"]].to_string(index=False))
_multi = AUDIT[AUDIT["n_distinct_boxes_after_think"] > 1]
print(f"\n--- 複数の異なる box を持つ行: {len(_multi)}（最後の box が参照正解と一致した行数: {(int((_multi['verify_status'] == 'match').sum()))}）例 ---")
print(_multi[["row", "n_boxes_after_think", "n_distinct_boxes_after_think", "first_box_after_think", "teacher_answer", "answer", "verify_status"]].head(8).to_string(index=False))

AO_DATA_READY = (len(_failed) == 0) and (len(_unacknowledged) == 0)
if not AO_DATA_READY:
    print("\n*** AO データに未解決の抽出/照合問題がある。同一データ条件の本学習へは進まない。"
          "不一致行を確認し、教師の最終回答として保持するなら AO_ACKNOWLEDGED_VERIFY_MISMATCH_ROWS に行番号を追加する ***")
else:
    print("\nAO_DATA_READY = True（抽出失敗 0、未確認の不一致 0）")

# ---- 3.4 派生データの保存（元テーブルに列を追加するだけ。行順・message・advantage はそのまま） ----
AO_TABLE = RAW_TABLE
for _name, _vals, _type in [("teacher_answer", _targets, pa.string()), ("ao_extraction_status", _status, pa.string()),
                            ("ao_verify_status", _verify, pa.string()), ("ao_source_row", list(range(len(_targets))), pa.int64()),
                            ("ao_n_boxes_after_think", AUDIT["n_boxes_after_think"].tolist(), pa.int64())]:
    AO_TABLE = AO_TABLE.append_column(_name, pa.array(_vals, type=_type))
# 抽出失敗行は teacher_answer=null のまま保存する（削除も answer 代用もしない）。ゲートで本学習を止める。
assert AO_TABLE.num_rows == RAW_TABLE.num_rows
assert AO_TABLE.column("message").equals(RAW_TABLE.column("message")) and AO_TABLE.column("advantage").equals(RAW_TABLE.column("advantage"))
AO_PARQUET = f"{DATA_DIR}/Math-AO-20k.parquet"
pq.write_table(AO_TABLE, AO_PARQUET)
AO_SHA256 = sha256_of(AO_PARQUET)
# 読み戻して元列が同一であることを確認
_rt = pq.read_table(AO_PARQUET)
for _c in RAW_TABLE.column_names:
    assert _rt.column(_c).equals(RAW_TABLE.column(_c)), f"column {_c} changed after round trip"
print("saved AO parquet:", AO_PARQUET, "sha256:", AO_SHA256)

AUDIT.to_csv(f"{RECORD_DIR}/ao_audit_rows.csv", index=False)
with open(f"{RECORD_DIR}/ao_audit_summary.json", "w") as f:
    json.dump(dict(_summary, ao_parquet=AO_PARQUET, ao_sha256=AO_SHA256, ao_data_ready=AO_DATA_READY,
                   acknowledged_verify_mismatch_rows=AO_ACKNOWLEDGED_VERIFY_MISMATCH_ROWS,
                   extraction_method="verl.utils.reward_score.math_verify_ours.last_boxed_only_string on text after </think>; "
                                     "verification = math_verify_ours.compute_score(target, answer)"), f, indent=2, ensure_ascii=False)
_flag_rows = sorted(set(_failed["row"]) | set(_mismatch["row"]) | set(_multi["row"].head(200)))
with open(f"{RECORD_DIR}/ao_flagged_rows.jsonl", "w") as f:
    for _r in _flag_rows:
        _rec = {k: (v.item() if hasattr(v, "item") else v) for k, v in AUDIT.iloc[_r].to_dict().items()}
        _rec["response_tail"] = _raw_df["response"].iloc[_r][-400:]
        f.write(json.dumps(_rec, ensure_ascii=False, default=str) + "\n")
print("audit files:", f"{RECORD_DIR}/ao_audit_summary.json", f"{RECORD_DIR}/ao_audit_rows.csv", f"{RECORD_DIR}/ao_flagged_rows.jsonl")
AO_AUDIT_SUMMARY = _summary
