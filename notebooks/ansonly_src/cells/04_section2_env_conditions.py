# @title 2. 環境の記録・HF ログイン・1 GPU の FSDP2 勾配確認・研究条件
import json, os, sys, platform, shutil, subprocess, time
import huggingface_hub

# ---- 2.1 セル 1 の結果（公式環境・Fork の revision・GPU）を記録する ------------------------------------------
FORK_REPO_URL, FORK_REF, FORK_COMMIT = ENV_BOOTSTRAP["fork_repo"], ENV_BOOTSTRAP["fork_ref"], ENV_BOOTSTRAP["fork_commit"]
UPSTREAM_REPO_URL, UPSTREAM_REFERENCE_COMMIT = ENV_BOOTSTRAP["upstream_repo"], ENV_BOOTSTRAP["upstream_reference_commit"]
UPSTREAM_FETCHED = ENV_BOOTSTRAP["upstream_fetched"]
OFFICIAL_CODE_UNCHANGED, OFFICIAL_DIFF_STAT = ENV_BOOTSTRAP["official_code_unchanged"], ENV_BOOTSTRAP["official_diff_stat"]
GPUS, TORCH_PROBE, PKG_VERSIONS = ENV_BOOTSTRAP["gpus"], ENV_BOOTSTRAP["torch_probe"], dict(ENV_BOOTSTRAP["packages"])
N_GPUS = len(GPUS)
GPU_PROFILE, GPU_PROFILE_DEVIATION = ENV_BOOTSTRAP["gpu_profile"], ENV_BOOTSTRAP["gpu_profile_deviation"]
FLASH_ATTN_VERSION = ENV_BOOTSTRAP["flash_attn_version"]
FLASH_ATTN_OK = bool(TORCH_PROBE.get("cuda_available")) and PKG_VERSIONS.get("flash_attn") == FLASH_ATTN_VERSION
assert platform.python_version() == TORCH_PROBE["python"] and sys.prefix == ENV_BOOTSTRAP["train_env_dir"], \
    f"このセルは公式環境のカーネルで実行する（%%ao）。python={platform.python_version()} prefix={sys.prefix}"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

_mem = {}
for line in open("/proc/meminfo"):
    _mem[line.split(":")[0]] = int(line.split(":")[1].strip().split()[0])
_disk = shutil.disk_usage(WORK_DIR)
print(f"GPU x{N_GPUS}: {GPUS} | profile={GPU_PROFILE}")
print(f"RAM total {_mem.get('MemTotal', 0) / 1e6:.1f} GB | disk free {_disk.free / 1e9:.1f} GB | kernel python {ENV_BOOTSTRAP['kernel_python']} | 公式環境 python {TORCH_PROBE['python']}")
print(f"torch {TORCH_PROBE['torch']} (CUDA {TORCH_PROBE['cuda']}, cuDNN {TORCH_PROBE['cudnn']}, NCCL {TORCH_PROBE['nccl']}) | flash-attn {PKG_VERSIONS.get('flash_attn')} | transformers {PKG_VERSIONS.get('transformers')}")
if GPU_PROFILE_DEVIATION:
    print(f"*** 公式環境からの変更（GPU {GPUS[0]['name']} のため）: {GPU_PROFILE_DEVIATION} ***")
print("公式コード (verl/, training_scripts/) の参照 commit からの差分:", "なし" if OFFICIAL_CODE_UNCHANGED else OFFICIAL_DIFF_STAT)

# ---- 2.2 HF ログイン（公式環境の huggingface_hub で行う。アカウント名は whoami から取る） -------------------------
HF_LOGGED_IN, HF_ACCOUNT_NAME = False, None   # GitHub のユーザー名からは推測しない
if os.environ.get("HF_TOKEN"):
    huggingface_hub.login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    HF_ACCOUNT_NAME = huggingface_hub.whoami().get("name")
    HF_LOGGED_IN = True
    print(f"HF: logged in as '{HF_ACCOUNT_NAME}' (token masked) | huggingface_hub {huggingface_hub.__version__}")
else:
    print("HF: HF_TOKEN が無い（セル 0）。HF へのアップロード・再開は無効")

# ---- 2.3 1 GPU の FSDP2 で勾配が正しく計算されるかの確認（GPU があるときだけ） ----------------------------------
# 1 GPU では FSDP2 の reduce-scatter が world size 1 の NCCL ReduceOp.AVG になる。torch には「world size 1 の AVG で一部の勾配が 0 になる」
# 未解決の報告がある（pytorch/pytorch#165399, #144045）。公式の apply_fsdp2 で包んだ小型 Qwen3 の勾配を、FSDP 無しの同じモデルの勾配と比べる。
_GRAD_CHECK_SRC = r'''
import json, os, copy, torch, torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import Qwen3Config, AutoModelForCausalLM
from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_load_full_state_dict, MixedPrecisionPolicy

cuda = torch.cuda.is_available()
if not cuda and os.environ.get("AO_GRADCHECK_CPU_PATCH") == "1":   # CPU での検証用（torch 2.6 の CPU Stream に FSDP2 が使うメソッドが無い）
    torch.cpu.Stream.record_event = lambda self, event=None: torch.cpu.Event()
    torch.cpu.Stream.wait_event = lambda self, event: None
dist.init_process_group("nccl" if cuda else "gloo")
dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if cuda else torch.device("cpu")
if cuda:
    torch.cuda.set_device(dev)
# Qwen3-1.7B と同じ語彙・tied embedding の小型モデル（重みは乱数。ダウンロードしない）
cfg = Qwen3Config(vocab_size=151936, hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=64, tie_word_embeddings=True, max_position_embeddings=1024)
torch.manual_seed(0)
model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
ref = copy.deepcopy(model).to(dev)
for m in (model, ref):   # 公式 trainer と同じ（enable_gradient_checkpointing=True）
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
mesh = init_device_mesh(dev.type, mesh_shape=(dist.get_world_size(),), mesh_dim_names=("fsdp",))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True)   # 公式 trainer と同じ
full_state = model.state_dict()
apply_fsdp2(model, dict(mesh=mesh, mp_policy=mp, offload_policy=None, reshard_after_forward=True), {})
fsdp2_load_full_state_dict(model, full_state, mesh, None)
g = torch.Generator().manual_seed(1)
for _ in range(2):   # 公式と同じく micro batch ごとに backward して勾配を累積する
    ids = torch.randint(0, cfg.vocab_size, (4, 128), generator=g).to(dev)
    for m in (model, ref):
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
            logits = m(input_ids=ids).logits
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, cfg.vocab_size), ids[:, 1:].reshape(-1))
        loss.backward()
ref_params = dict(ref.named_parameters())
rows = []
for name, p in model.named_parameters():
    gf = p.grad.full_tensor().float().cpu()
    gr = ref_params[name].grad.float().cpu()
    rel = float((gf - gr).norm() / gr.norm().clamp_min(1e-30))
    zeroed = int(((gr != 0) & (gf == 0)).sum())
    rows.append(dict(name=name, rel=rel, zeroed=zeroed, numel=gr.numel(), tail_ref=gr.flatten()[-2:].tolist(), tail_fsdp=gf.flatten()[-2:].tolist()))
ok = all(r["rel"] < 1e-2 and r["zeroed"] == 0 for r in rows)
if dist.get_rank() == 0:
    print("GRADCHECK " + json.dumps(dict(ok=ok, world_size=dist.get_world_size(), torch=torch.__version__,
                                         nccl=list(torch.cuda.nccl.version()) if cuda else None, params=rows)))
dist.destroy_process_group()
'''


def run_fsdp2_grad_check(nproc=1, extra_env=None, timeout=900):
    path = f"{WORK_DIR}/tools/fsdp2_grad_check.py"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(_GRAD_CHECK_SRC)
    env = dict(os.environ, PYTHONPATH=REPO_DIR, **(extra_env or {}))
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", path]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO_DIR, timeout=timeout)
    lines = [l for l in r.stdout.splitlines() if l.startswith("GRADCHECK ")]
    if r.returncode != 0 or not lines:
        return dict(status="error", returncode=r.returncode, stderr=r.stderr[-3000:])
    res = json.loads(lines[-1][len("GRADCHECK "):])
    res["status"] = "ok" if res["ok"] else "mismatch"
    return res


FSDP2_GRAD_CHECK = dict(status="skipped (GPU なし)")
if N_GPUS >= 1:
    # 公式スクリプトの export（セクション 5 で表示）と同じ環境変数で実行する（同期の違いで結果が変わる不具合を見逃さないため）
    FSDP2_GRAD_CHECK = run_fsdp2_grad_check(nproc=1, extra_env=dict(CUDA_LAUNCH_BLOCKING="1", TORCH_NCCL_AVOID_RECORD_STREAMS="1", NCCL_DEBUG="WARN"))
    if FSDP2_GRAD_CHECK["status"] == "ok":
        _worst = max(FSDP2_GRAD_CHECK["params"], key=lambda r: r["rel"])
        print(f"FSDP2 勾配確認 (world size 1): OK。全 {len(FSDP2_GRAD_CHECK['params'])} パラメータで 0 化なし、最大相対差 {_worst['rel']:.2e} ({_worst['name']})")
    elif FSDP2_GRAD_CHECK["status"] == "mismatch":
        for _r in FSDP2_GRAD_CHECK["params"]:
            if _r["rel"] >= 1e-2 or _r["zeroed"]:
                print("  MISMATCH", _r)
        print("*** FSDP2 (world size 1) の勾配が FSDP 無しの勾配と一致しない。この環境では学習を開始しない（REQUIRE_FSDP2_GRAD_CHECK） ***")
    else:
        print("*** FSDP2 勾配確認のスクリプトが失敗した ***\n", FSDP2_GRAD_CHECK.get("stderr", ""))
else:
    print("FSDP2 勾配確認: GPU が無いのでスキップ")

ENV_RECORD = dict(ENV_BOOTSTRAP, session_started_at=SESSION_STARTED_AT, n_gpus=N_GPUS, ram_total_gb=round(_mem.get("MemTotal", 0) / 1e6, 1),
                  disk_free_gb=round(_disk.free / 1e9, 1), flash_attn_ok=FLASH_ATTN_OK, hf_account=HF_ACCOUNT_NAME,
                  fsdp2_grad_check={k: v for k, v in FSDP2_GRAD_CHECK.items() if k != "params"})
with open(f"{RECORD_DIR}/env_record.json", "w") as f:
    json.dump(ENV_RECORD, f, indent=2, ensure_ascii=False)
print("saved", f"{RECORD_DIR}/env_record.json")

# ---- 2.4 研究条件・対応モデル -------------------------------------------------------------------
print("\n=== 対応モデル ===")
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
