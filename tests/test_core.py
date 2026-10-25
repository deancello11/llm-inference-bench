import math

from bench.config import load_matrix
from bench.stats import binomial_stderr, pareto_frontier, percentile, trapezoid_energy_j
from bench.workloads import make_prompts, seed_for


def test_matrix_loads_and_ids_unique():
    m = load_matrix()
    assert {c.stack for c in m.configs} == {"hf", "vllm", "llamacpp"}
    assert m.workloads["short"].input_len == 128 and m.workloads["long"].input_len == 1024
    assert m.get("vllm-awq").quantization == "awq"
    assert all(c.file for c in m.configs if c.stack == "llamacpp")


def test_prompts_exact_length_distinct_deterministic():
    a = make_prompts(128, 4, seed_for("short", 4, 0, 0, 0))
    b = make_prompts(128, 4, seed_for("short", 4, 0, 0, 0))
    c = make_prompts(128, 4, seed_for("short", 4, 0, 0, 1))
    assert a == b and a != c
    assert all(len(p) == 128 for p in a)
    assert len({tuple(p) for p in a}) == 4          # distinct => no prefix-cache hits


def test_trapezoid_constant_power():
    assert math.isclose(trapezoid_energy_j([0, 1, 2, 4], [100, 100, 100, 100]), 400.0)


def test_trapezoid_ramp():
    assert math.isclose(trapezoid_energy_j([0, 2], [0, 100]), 100.0)


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert math.isclose(percentile([0, 10], 95), 9.5)


def test_pareto_frontier_min_x_max_y():
    pts = [(1, 50), (2, 70), (3, 60), (4, 80), (2, 70)]
    # (3,60) dominated by (2,70); the duplicate (2,70) points don't dominate each other
    assert pareto_frontier(pts) == [0, 1, 4, 3]


def test_binomial_stderr():
    assert math.isclose(binomial_stderr(0.5, 100), 0.05)
