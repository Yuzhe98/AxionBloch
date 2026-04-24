# COMMAND TO RUN THIS TEST:
# $env:NUMPY_PERF_PRINT_RESULTS="1"; pytest tests/test_numpy_unit_performance.py -q -s
import os
import time

import numpy as np
import pytest
from astropy import units as u


ARRAY_SIZE = int(os.getenv("NUMPY_PERF_ARRAY_SIZE", "1000000"))
REPEAT = int(os.getenv("NUMPY_PERF_REPEAT", "10"))
PRINT_RESULTS = os.getenv("NUMPY_PERF_PRINT_RESULTS", "0") == "1"
SEED = int(os.getenv("NUMPY_PERF_SEED", "42"))


def _timed_average(callable_obj, repeat=REPEAT):
    """Return average runtime in milliseconds for repeated calls."""
    callable_obj()  # warm-up call
    durations = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        callable_obj()
        durations.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(durations))


def _run_numpy_operation(array_like, operation):
    """Run a selected operation on a NumPy array or an Astropy Quantity array."""
    if operation == "addition":
        return array_like + array_like
    if operation == "multiplication":
        return array_like * 2.5
    if operation == "sqrt":
        return np.sqrt(array_like)
    if operation == "sum":
        return np.sum(array_like)
    if operation == "mean":
        return np.mean(array_like)
    if operation == "std":
        return np.std(array_like)
    if operation == "unit_conversion":
        if not hasattr(array_like, "to"):
            pytest.skip("Unit conversion requires an Astropy Quantity array.")
        return array_like.to(u.cm * u.microsecond * u.g)
    raise ValueError(f"Unsupported operation: {operation}")


def _assert_result_valid(result, operation, use_units):
    """Validate result shape/type without constraining benchmark timings."""
    if operation in {"sum", "mean", "std"}:
        value = result.value if use_units else result
        assert value > 0
        return
    if operation == "unit_conversion":
        assert use_units
        assert result.unit == u.cm * u.microsecond * u.g
        return
    assert result is not None


def compare_numpy_operation_efficiency(
    array_size=ARRAY_SIZE, repeat=REPEAT, print_results=False, seed=SEED
):
    """Compare timing for plain NumPy arrays vs Astropy Quantity arrays."""
    rng = np.random.default_rng(seed)
    arr = rng.random(array_size)
    arr_with_unit = arr * u.m * u.s * u.kg  # attach a composite unit for more realistic conversion
    operations = (
        "addition",
        "multiplication",
        "sqrt",
        "sum",
        "mean",
        "std",
        "unit_conversion",
    )

    results = {}
    for operation in operations:
        op_results = {
            "plain_ms": None,
            "with_units_ms": None,
            "ratio_units_over_plain": None,
        }

        if operation != "unit_conversion":
            plain_result = _run_numpy_operation(arr, operation)
            _assert_result_valid(plain_result, operation, use_units=False)
            op_results["plain_ms"] = _timed_average(
                lambda: _run_numpy_operation(arr, operation), repeat
            )

        unit_result = _run_numpy_operation(arr_with_unit, operation)
        _assert_result_valid(unit_result, operation, use_units=True)
        op_results["with_units_ms"] = _timed_average(
            lambda: _run_numpy_operation(arr_with_unit, operation), repeat
        )

        plain_ms = op_results["plain_ms"]
        if plain_ms is not None and plain_ms > 0:
            op_results["ratio_units_over_plain"] = (
                op_results["with_units_ms"] / plain_ms
            )

        results[operation] = op_results

    if print_results:
        print("\nNumPy vs Astropy Quantity timing comparison (ms)")
        print(f"array_size={array_size:,}, repeat={repeat}, seed={seed}")
        print("operation         plain (ms)    with_units (ms)    runtime ratio (units/plain)")
        for operation, values in results.items():
            plain = "N/A" if values["plain_ms"] is None else f"{values['plain_ms']:.3f}"
            with_units = f"{values['with_units_ms']:.3f}"
            ratio = (
                "N/A"
                if values["ratio_units_over_plain"] is None
                else f"{values['ratio_units_over_plain']:.2f}x"
            )
            print(f"{operation:<16} {plain:>9}    {with_units:>13}    {ratio:>18}")

    return results


def test_numpy_operation_efficiency_comparison():
    """Run one function that benchmarks all operations and validates timings."""
    results = compare_numpy_operation_efficiency(print_results=PRINT_RESULTS)

    assert results["addition"]["plain_ms"] > 0
    assert results["addition"]["with_units_ms"] > 0
    assert results["multiplication"]["plain_ms"] > 0
    assert results["multiplication"]["with_units_ms"] > 0
    assert results["sqrt"]["plain_ms"] > 0
    assert results["sqrt"]["with_units_ms"] > 0
    assert results["sum"]["plain_ms"] > 0
    assert results["sum"]["with_units_ms"] > 0
    assert results["mean"]["plain_ms"] > 0
    assert results["mean"]["with_units_ms"] > 0
    assert results["std"]["plain_ms"] > 0
    assert results["std"]["with_units_ms"] > 0
    assert results["unit_conversion"]["plain_ms"] is None
    assert results["unit_conversion"]["with_units_ms"] > 0
