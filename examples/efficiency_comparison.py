import time
import numpy as np
import sympy
import symengine as se

# Import your pybind11 C++ module
import blochSimulation_c_ext.blochSimulation_c as bh

# ----------------------------
# Setup: evaluation points
# ----------------------------
N = 1_000_000
N = 1_000_000
r_vals = np.logspace(0.1, 10, N)  # large array for benchmarking

amp = 2.1341897
a = 1.13124

# ----------------------------
# Dictionary to store runtimes
# ----------------------------
runtimes = {}

# ----------------------------
# 1️⃣ SymPy
# ----------------------------
r_sym = sympy.symbols("r")
expr_sympy = amp / ((1 + r_sym/a) * (1 + (r_sym/a)**2))

# Plain SymPy
f_sympy = sympy.lambdify(r_sym, expr_sympy)
t0 = time.time()
val_sympy = f_sympy(r_vals)
t1 = time.time()
runtimes["SymPy plain"] = t1-t0
print(f"SymPy plain: {t1-t0:.6f} s")

# SymPy + NumPy
f_sympy_numpy = sympy.lambdify(r_sym, expr_sympy, modules="numpy")
t0 = time.time()
val_sympy_np = f_sympy_numpy(r_vals)
t1 = time.time()
runtimes["SymPy+NumPy"] = t1-t0
print(f"SymPy+NumPy: {t1-t0:.6f} s")

# SymPy + NumExpr
f_sympy_numexpr = sympy.lambdify(r_sym, expr_sympy, modules="numexpr")
t0 = time.time()
val_sympy_ne = f_sympy_numexpr(r_vals)
t1 = time.time()
runtimes["SymPy+NumExpr"] = t1-t0
print(f"SymPy+NumExpr: {t1-t0:.6f} s")

# ----------------------------
# 2️⃣ SymEngine
# ----------------------------
r_se = se.Symbol("r")
expr_se = amp / ((1 + r_se/a) * (1 + (r_se/a)**2))
f_se = se.Lambdify([r_se], expr_se)

t0 = time.time()
val_se = f_se(r_vals)
t1 = time.time()
runtimes["SymEngine"] = t1-t0
print(f"SymEngine vectorized: {t1-t0:.6f} s")

# ----------------------------
# 3️⃣ NumPy
# ----------------------------
t0 = time.time()
val_np = amp / ((1 + r_vals/a) * (1 + (r_vals/a)**2))
t1 = time.time()
runtimes["NumPy"] = t1-t0
print(f"NumPy numerical: {t1-t0:.6f} s")

# ----------------------------
# 4️⃣ C++ via pybind11
# ----------------------------
t0 = time.time()
val_cpp = bh.burkert_potential_vector(r_vals, amp, a)
t1 = time.time()
runtimes["C++ pybind11"] = t1-t0
print(f"C++ pybind11: {t1-t0:.6f} s")

# ----------------------------
# Verify outputs
# ----------------------------
val_cpp_np = np.array(val_cpp)
assert np.allclose(val_np, val_cpp_np, rtol=1e-12), "C++ module does not match NumPy output."

# ----------------------------
# 5️⃣ Summary table
# ----------------------------
print("\n=== Benchmark Summary ===")
fastest_time = min(runtimes.values())

for name, t in runtimes.items():
    rel = t / fastest_time
    print(f"{name:<20}: {t:.6f} s  |  relative: {rel:.2f}x")

