"""Orakel (ORACLE-PLAYBOOK): jedes Verfahren gegen einen anderen Rechenweg.

* Varianzverhältnis der gestutzten Normalverteilung gegen `scipy.stats.truncnorm`; Thorndike- und Summenformel gegen große Simulationen und gegen numerische Integration.
* Varianz des Fenstermittels der Welle gegen die volle Toeplitz-Kovarianzmatrix (`scipy.linalg.toeplitz`); die erwarteten Korrelationen (gesamt, Fenster, zwischen, innerhalb, innerhalb mit Fenstern)
  gegen Simulationen der Welt.
* Zerlegung gegen die Schleife über Standorte; Pearson gegen `numpy.corrcoef` und `scipy.stats.pearsonr`; Verdünnung gegen die Simulation.
Die Bibliotheken stehen nur im Test-venv; das Orakel braucht sie nicht zur Laufzeit."""

import math

import numpy as np
import pytest

import agr_algorithm as K
import prs_scenario as S

scipy_stats = pytest.importorskip("scipy.stats")
scipy_linalg = pytest.importorskip("scipy.linalg")
scipy_integrate = pytest.importorskip("scipy.integrate")


@pytest.mark.parametrize("q", [0.9, 0.5, 0.2, 0.1, 0.05, 0.01])
def test_truncation_variance_matches_scipy(q):
    c = scipy_stats.norm.ppf(1 - q)
    assert K.trunc_variance_ratio(q) == pytest.approx(float(scipy_stats.truncnorm(c, math.inf).var()), rel=1e-9)


@pytest.mark.parametrize("rho", [0.3, 0.5, 0.8])
@pytest.mark.parametrize("q", [0.5, 0.1])
def test_selection_formulas_match_a_large_simulation(rho, q):
    x, y = K.draw_pairs(1_000_000, rho, S.VRng(11))
    top_x = np.argsort(-x)[: int(len(x) * q)]
    assert K.pearson_r(x[top_x], y[top_x]) == pytest.approx(K.exact_select_x(rho, q), abs=0.004)
    top_s = K.select_top(x, y, q, "summe")
    assert K.pearson_r(x[top_s], y[top_s]) == pytest.approx(K.exact_select_sum(rho, q), abs=0.008)


@pytest.mark.parametrize("rho,q", [(0.5, 0.5), (0.5, 0.1), (0.8, 0.2)])
def test_selection_by_x_matches_numerical_integration(rho, q):
    """Korrelation in der Menge X > c: Momente per Integration der Dichte (Zerlegung Y = ρX + √(1 − ρ²) Z)."""
    c = scipy_stats.norm.ppf(1 - q)
    pdf = scipy_stats.norm.pdf
    m0 = scipy_integrate.quad(pdf, c, math.inf)[0]
    m1 = scipy_integrate.quad(lambda v: v * pdf(v), c, math.inf)[0] / m0
    m2 = scipy_integrate.quad(lambda v: v * v * pdf(v), c, math.inf)[0] / m0
    var_x = m2 - m1 * m1
    cov_xy = rho * var_x
    var_y = rho * rho * var_x + 1 - rho * rho
    assert K.exact_select_x(rho, q) == pytest.approx(cov_xy / math.sqrt(var_x * var_y), abs=1e-6)


@pytest.mark.parametrize("m,smooth", [(1, 30), (7, 30), (30, 30), (90, 30), (7, 7), (30, 7), (5, 1)])
def test_wave_mean_variance_matches_the_toeplitz_covariance_matrix(m, smooth):
    acov = np.maximum(0.0, 1.0 - np.arange(m) / smooth)
    cov = scipy_linalg.toeplitz(acov)
    assert K.wave_mean_variance(m, smooth) == pytest.approx(float(np.ones(m) @ cov @ np.ones(m)) / (m * m), rel=1e-12)


@pytest.mark.parametrize("m", [1, 7, 30])
def test_the_pure_window_formula_matches_a_simulation(m):
    T, runs = 3600, 40
    rng = S.VRng(21 + m)
    rs = []
    for _ in range(runs):
        w = K.wave(1, T, 30, rng)[0]
        a, b = w + 2.0 * rng.normal(T), w + 2.0 * rng.normal(T)
        rs.append(K.pearson_r(K.window_means(a, m), K.window_means(b, m)))
    assert float(np.mean(rs)) == pytest.approx(K.exact_wave_window_r(m, 30, 2.0), abs=0.02)


@pytest.mark.parametrize("smooth,sigma,gamma,rho_b", [(30, 2.0, 1.0, 0.5), (7, 1.0, -0.5, 0.8), (1, 1.0, 0.5, -0.3)])
def test_total_between_and_within_expectations_match_a_world_simulation(smooth, sigma, gamma, rho_b):
    G, T = 100, 100
    rng = S.VRng(31 + smooth)
    tot, bet, wit = [], [], []
    for _ in range(60):
        X, Y, _a, _b = K.make_world(G, T, rho_b, sigma, smooth, gamma, rng)
        s = K.split_levels(X, Y)
        tot.append(s["total"])
        bet.append(s["between"])
        wit.append(s["within"])
    assert float(np.mean(tot)) == pytest.approx(K.exact_pooled_r(1, smooth, sigma, gamma, rho_b), abs=0.02)
    assert float(np.mean(bet)) == pytest.approx(K.exact_pooled_r(T, smooth, sigma, gamma, rho_b), abs=0.04)
    assert float(np.mean(wit)) == pytest.approx(K.exact_within_r(T, smooth, sigma, gamma), abs=0.02)


@pytest.mark.parametrize("m", [1, 5, 20])
def test_the_within_window_expectation_matches_a_world_simulation(m):
    G, T, smooth, sigma, gamma = 120, 120, 30, 1.0, 1.0
    rng = S.VRng(41 + m)
    vals = []
    for _ in range(50):
        X, Y, _a, _b = K.make_world(G, T, 0.0, sigma, smooth, gamma, rng)
        dx, dy = X - X.mean(axis=1, keepdims=True), Y - Y.mean(axis=1, keepdims=True)
        vals.append(K.r_all(K.window_means(dx, m), K.window_means(dy, m)))
    assert float(np.mean(vals)) == pytest.approx(K.exact_within_window_r(T, m, smooth, sigma, gamma), abs=0.02)


def test_split_levels_matches_the_loop_over_sites():
    X, Y, _a, _b = K.make_world(7, 30, 0.5, 1.0, 7, -0.5, S.VRng(5))
    s = K.split_levels(X, Y)
    within = sum(float(np.sum((X[g] - X[g].mean()) * (Y[g] - Y[g].mean()))) for g in range(7))
    between = 30 * sum(float((X[g].mean() - X.mean()) * (Y[g].mean() - Y.mean())) for g in range(7))
    assert s["ss_within"] == pytest.approx(within, rel=1e-12) and s["ss_between"] == pytest.approx(between, rel=1e-12)
    assert s["total"] == pytest.approx(float(np.corrcoef(X.ravel(), Y.ravel())[0, 1]), abs=1e-12) and s["between"] == pytest.approx(float(np.corrcoef(X.mean(1), Y.mean(1))[0, 1]), abs=1e-12)


def test_pearson_copy_matches_numpy_and_scipy():
    x, y = K.draw_pairs(500, 0.4, S.VRng(8))
    assert K.pearson_r(x, y) == pytest.approx(float(np.corrcoef(x, y)[0, 1]), abs=1e-12) and K.pearson_r(x, y) == pytest.approx(scipy_stats.pearsonr(x, y)[0], abs=1e-12)


@pytest.mark.parametrize("rx,ry", [(0.8, 0.8), (0.5, 0.8), (0.4, 0.4)])
def test_attenuation_matches_a_simulation_and_the_correction_is_unbiased(rx, ry):
    rs = []
    for i in range(40):
        a, b = K.noisy_pairs(2000, 0.6, rx, ry, S.VRng(100 + i))
        rs.append(K.pearson_r(a, b))
    assert float(np.mean(rs)) == pytest.approx(K.attenuated_r(0.6, rx, ry), abs=0.01)
    assert float(np.mean([K.disattenuate(r, rx, ry) for r in rs])) == pytest.approx(0.6, abs=0.015)
