"""Randfälle (vor dem ersten Mutationstest): Reglergrenzen, inklusive Grenzen der Fenster und der Auswahl, konstante Werte, kleinste Welt."""

import numpy as np
import pytest

import agr_algorithm as K
import agr_constants as C
import agr_evaluation as ev
import prs_scenario as S


@pytest.mark.parametrize("edge", ["min", "max"])
def test_analyse_runs_at_the_corners_of_every_slider(edge):
    lo = edge == "min"
    s = ev.Settings(C.G_MIN if lo else C.G_MAX, C.T_MIN if lo else C.T_MAX, C.RHO_B_MIN if lo else C.RHO_B_MAX, C.SIGMA_MIN if lo else C.SIGMA_MAX, C.SMOOTHS[0] if lo else C.SMOOTHS[-1],
                    C.GAMMA_MIN if lo else C.GAMMA_MAX, C.WINDOWS[-1] if lo else C.WINDOWS[0], "summe" if lo else "x", C.Q_MIN if lo else C.Q_MAX, 3)
    a = ev.analyse(s)
    assert all(np.isfinite(r.r) for r in a.rows) and all(r.exact is None or -1.0 <= r.exact <= 1.0 for r in a.rows)


def test_the_window_needs_at_least_two_full_windows_per_site():
    """Inklusive Grenze T = 2 m: genau zwei Fenster je Standort; bei T < 2 m ist die Abweichung vom Standortmittel im einzigen Fenster null und die Zeile entfällt."""
    ok = ev.analyse(ev.Settings(g=10, t=180, window=90))
    assert ev.row(ok, "window").points == 20 and np.isfinite(ev.row(ok, "window").r) and np.isfinite(ev.row(ok, "window").exact)
    short = ev.analyse(ev.Settings(g=10, t=179, window=90))
    assert ev.row(short, "window") is None
    kind, text = ev.verdict_window(short)
    assert kind == "warning" and "weniger als zwei volle Fenster" in text and "179 Tagen" in text
    assert ev.row(ev.analyse(ev.Settings(g=10, t=90, window=90)), "window") is None
    assert K.window_means(np.arange(90.0)[None, :], 90).tolist() == [[44.5]]
    with pytest.raises(ValueError):
        K.window_means(np.arange(90.0)[None, :], 91)
    assert np.isfinite(K.exact_within_window_r(180, 90, 30, 1.0, 1.0))
    for t_, m in ((179, 90), (90, 90), (100, 101)):
        with pytest.raises(ValueError):
            K.exact_within_window_r(t_, m, 30, 1.0, 1.0)


def test_the_selection_always_keeps_at_least_three_sites():
    a = ev.analyse(ev.Settings(g=30, select="x", q=5))
    assert len(a.selected) == C.MIN_SELECTED and ev.row(a, "selected").points == C.MIN_SELECTED
    assert K.select_count(30, 0.05) == 3 and K.select_count(30, 0.1) == 3 and K.select_count(30, 0.11) == 3 and K.select_count(30, 0.12) == 4


def test_full_selection_is_the_whole_set_and_leaves_r_unchanged():
    a = ev.analyse(ev.Settings(g=12, select="x", q=100))
    assert len(a.selected) == 12 and ev.row(a, "selected").r == pytest.approx(ev.row(a, "between").r, abs=1e-12)
    assert ev.row(a, "selected").exact == pytest.approx(ev.row(a, "between").exact, abs=1e-12)


def test_selection_by_the_sum_with_a_constant_variable_does_not_break():
    x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    y = np.full(5, 7.0)
    assert list(K.select_top(x, y, 0.4, "summe")) == [3, 4, 1] and list(K.standardize(y)) == [0.0] * 5


def test_without_noise_there_is_no_corrected_row_and_the_reliabilities_are_one():
    a = ev.analyse(ev.Settings(sigma=0.0))
    assert ev.row(a, "corrected") is None and (a.rx, a.ry) == (1.0, 1.0) and ev.row(a, "total").r == pytest.approx(ev.row(a, "true").r, abs=1e-12)


def test_the_smallest_world_runs():
    X, Y, _a, _b = K.make_world(2, 2, 0.0, 1.0, 1, 1.0, S.VRng(1))
    assert X.shape == (2, 2) and np.isfinite(K.split_levels(X, Y)["total"])


def test_the_exact_selection_formulas_accept_a_quota_of_one_and_reject_zero():
    assert K.exact_select_x(0.6, 1.0) == pytest.approx(0.6) and K.exact_select_sum(0.6, 1.0) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        K.exact_select_x(0.6, 0.0)
    assert K.select_count(10, 1.0) == 10 and K.select_count(10, 0.0) == C.MIN_SELECTED


def test_rho_b_at_the_slider_ends_gives_finite_exact_values():
    for rho in (C.RHO_B_MIN, C.RHO_B_MAX):
        assert -1.0 < K.exact_pooled_r(360, 30, 2.0, 1.0, rho) < 1.0 and -1.0 < K.exact_pooled_r(1, 1, 0.0, -1.0, rho) < 1.0


def test_analyse_with_a_negative_coupling_has_a_negative_within_correlation():
    a = ev.analyse(ev.Settings(rho_b=0.8, sigma=1.0, smooth=1, gamma=-0.5))
    assert ev.row(a, "within").r < 0 < ev.row(a, "between").r and ev.row(a, "within").exact < 0
