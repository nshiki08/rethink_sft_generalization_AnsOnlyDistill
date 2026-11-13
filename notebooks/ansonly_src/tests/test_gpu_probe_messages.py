"""CPU test of the GPU error messages of cell 1 (02_env_bootstrap.py). No GPU, no network, any Python 3.

Cell 1 as a whole needs a Colab-like kernel (test_env_bootstrap.py). Here only the two pure functions that decide the message
(_arch_ok, gpu_probe_problem) are taken out of the cell source and run with made-up probe results:
a CUDA initialization failure and a missing GPU architecture must give different messages.
Usage: python notebooks/ansonly_src/tests/test_gpu_probe_messages.py
"""
import ast, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
tree = ast.parse(open(os.path.join(SRC, "cells", "02_env_bootstrap.py"), encoding="utf-8").read())
funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_arch_ok", "gpu_probe_problem")]
assert sorted(f.name for f in funcs) == ["_arch_ok", "gpu_probe_problem"], [f.name for f in funcs]
g = {}
exec(compile(ast.Module(body=funcs, type_ignores=[]), "02_env_bootstrap.py", "exec"), g)
arch_ok, problem = g["_arch_ok"], g["gpu_probe_problem"]

G4 = [dict(index=0, name="NVIDIA RTX PRO 6000 Blackwell Server Edition", vram_mb=97887, driver="570.00", compute_cap="12.0")]
A100 = [dict(index=0, name="NVIDIA A100-SXM4-80GB", vram_mb=81920, driver="550.54.15", compute_cap="8.0")]
T26 = ["sm_50", "sm_60", "sm_70", "sm_80", "sm_86", "sm_90"]                 # torch 2.6.0 (cu124)
T27 = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]   # torch 2.7.1+cu128

# 1. アーキテクチャの判定（同じ major で minor が実機以下、または PTX）
assert arch_ok(T26, (8, 0)) and arch_ok(T26, (8, 9)) and arch_ok(T26, (9, 0))
assert not arch_ok(T26, (12, 0)) and arch_ok(T27, (12, 0)) and not arch_ok([], (12, 0))

# 2. GPU が無い / 問題が無い → None
ok_probe = dict(torch="2.7.1+cu128", cuda="12.8", cuda_available=True, arch_list=T27)
assert problem([], None, dict(torch="2.6.0+cpu", cuda=None, cuda_available=False, arch_list=[]), "official") is None
assert problem(G4, (12, 0), ok_probe, "blackwell") is None
assert problem(A100, (8, 0), dict(torch="2.6.0+cu124", cuda="12.4", cuda_available=True, arch_list=T26), "official") is None

# 3. GPU はあるが torch が CUDA を初期化できない → torch のエラーとドライバの版を出す。「GPU 用のコードが無い」とは言わない
init_err = "RuntimeError: The NVIDIA driver on your system is too old (found version 12040)."
m = problem(G4, (12, 0), dict(torch="2.7.1+cu128", cuda="12.8", cuda_available=False, arch_list=[], cuda_init_error=init_err), "blackwell")
assert m and "CUDA を初期化できない" in m and init_err in m and "570.00" in m and "2.7.1+cu128" in m and "12.8" in m, m
assert "用のコードが無い:" not in m and "用のコードが無い（" not in m, m
m_cpu = problem(A100, (8, 0), dict(torch="2.6.0+cpu", cuda=None, cuda_available=False, arch_list=[],
                                    cuda_init_error="AssertionError: Torch not compiled with CUDA enabled"), "official")
assert m_cpu and "CUDA を初期化できない" in m_cpu and "Torch not compiled with CUDA enabled" in m_cpu, m_cpu
m_none = problem(A100, (8, 0), dict(torch="2.6.0+cu124", cuda="12.4", cuda_available=False, arch_list=[]), "official")   # 旧い probe（エラー文なし）
assert m_none and "取得できなかった" in m_none

# 4. CUDA は使えるが、その GPU 用のコードが torch に無い（G4 に公式の torch 2.6.0）→ アーキテクチャの文
m2 = problem(G4, (12, 0), dict(torch="2.6.0+cu124", cuda="12.4", cuda_available=True, arch_list=T26), "official")
assert m2 and "用のコードが無い" in m2 and "(12, 0)" in m2 and "sm_90" in m2 and "official" in m2 and "CUDA を初期化できない" not in m2, m2
print("GPU PROBE MESSAGE TESTS PASSED")
