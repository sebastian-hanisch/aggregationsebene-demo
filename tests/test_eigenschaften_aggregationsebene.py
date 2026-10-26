"""Eigenschaften: Invarianzen, Symmetrien, Zerlegung, Monotonie der exakten Formeln, Reproduzierbarkeit."""

import numpy as np
import pytest

import agr_algorithm as K
import agr_evaluation as ev
import prs_scenario as S


def test_the_levels_are_invariant_under_shifting_and_scaling_and_symmetric_in_x_and_y():
    X, Y, _a, _b = K.make_world(10, 40, 0.5, 1.0, 7, 0.5, S.VRng(3))
    base = K.split_levels(X, Y)
    moved = K.split_levels(3.0 * X + 5.0, 0.25 * Y - 2.0)
    flipped = K.split_levels(-X, Y)
    swapped = K.split_levels(Y, X)
    for k in ("total", "between", "within"):
        assert moved[k] == pytest.approx(base[k], abs=1e-12) and flipped[k] == pytest.approx(-base[k], abs=1e-12) and swapped[k] == pytest.approx(base[k], abs=1e-12)


def test_the_decomposition_holds_in_many_random_worlds():
    rng = S.VRng(5)
    for _ in range(10):
        X, Y, _a, _b = K.make_world(8, 25, 0.3, 1.0, 7, -0.5, rng)
        s = K.split_levels(X, Y)
        assert s["ss_total"] == pytest.approx(s["ss_within"] + s["ss_between"], rel=1e-10, abs=1e-9)


def test_the_between_correlation_ignores_the_order_of_the_sites_and_days():
    X, Y, _a, _b = K.make_world(10, 40, 0.5, 1.0, 7, 1.0, S.VRng(3))
    g = np.random.default_rng(1).permutation(10)
    d = np.random.default_rng(2).permutation(40)
    a, b = K.split_levels(X, Y), K.split_levels(X[g][:, d], Y[g][:, d])
    for k in ("total", "between", "within"):
        assert b[k] == pytest.approx(a[k], abs=1e-12)


def test_complete_windows_keep_the_site_mean():
    X, _y, _a, _b = K.make_world(6, 60, 0.5, 1.0, 7, 1.0, S.VRng(3))
    for m in (1, 6, 30, 60):
        assert K.window_means(X, m).mean(axis=1) == pytest.approx(X.mean(axis=1), abs=1e-12)


def test_selection_by_x_does_not_change_under_a_monotone_transformation_and_by_sum_under_affine_scaling():
    x, y = K.draw_pairs(500, 0.5, S.VRng(7))
    assert set(K.select_top(x, y, 0.2, "x")) == set(K.select_top(np.exp(x), y, 0.2, "x"))
    assert set(K.select_top(x, y, 0.2, "summe")) == set(K.select_top(4.0 * x + 3.0, 0.5 * y - 1.0, 0.2, "summe"))


def test_the_exact_formulas_stay_in_range_and_are_monotone():
    for rho in (-0.8, -0.3, 0.0, 0.3, 0.8):
        for q in (1.0, 0.5, 0.1, 0.01):
            assert -1.0 <= K.exact_select_x(rho, q) <= 1.0 and -1.0 <= K.exact_select_sum(rho, q) <= 1.0
    for q in (0.5, 0.1):
        xs = [K.exact_select_x(r, q) for r in (-0.8, -0.3, 0.0, 0.3, 0.8)]
        ss = [K.exact_select_sum(r, q) for r in (-0.8, -0.3, 0.0, 0.3, 0.8)]
        assert xs == sorted(xs) and ss == sorted(ss)
    pooled = [K.exact_pooled_r(30, 30, 2.0, 1.0, r) for r in (-0.8, -0.3, 0.0, 0.5, 0.9)]
    assert pooled == sorted(pooled)


def test_more_noise_lowers_the_pooled_correlation_and_longer_windows_raise_it_with_memory():
    sig = [K.exact_pooled_r(1, 30, s, 1.0, 0.5) for s in (0.0, 1.0, 2.0, 3.0)]
    assert sig == sorted(sig, reverse=True)
    win = [K.exact_wave_window_r(m, 30, 2.0) for m in (1, 7, 30, 90)]
    assert win == sorted(win)


def test_the_same_seed_gives_the_same_world_and_a_different_seed_a_different_one():
    a, b = ev.analyse(ev.Settings(seed=1)), ev.analyse(ev.Settings(seed=1))
    assert [r.r for r in a.rows] == [r.r for r in b.rows]
    assert [r.r for r in a.rows] != [r.r for r in ev.analyse(ev.Settings(seed=2)).rows]


def test_the_expected_values_do_not_depend_on_the_seed():
    assert [r.exact for r in ev.analyse(ev.Settings(seed=1)).rows] == [r.exact for r in ev.analyse(ev.Settings(seed=2)).rows]
