"""Einheiten von Hand (DEMO-PLAYBOOK §1b): jede Funktion an einer Mini-Instanz, deren Ergebnis ohne Rechner folgt."""

import math

import numpy as np
import pytest

import agr_algorithm as K
import agr_constants as C
import prs_scenario as S


# --- Welle und Welt --------------------------------------------------------------------------------------------------------------------

def test_wave_without_memory_is_the_white_noise_itself():
    w = K.wave(3, 50, 1, S.VRng(4))
    z = S.VRng(4).normal((3, 50))
    assert w == pytest.approx(z, abs=1e-12)


def test_wave_has_unit_variance_and_the_triangular_autocovariance():
    w = K.wave(400, 300, 10, S.VRng(5))
    assert w.shape == (400, 300) and float(w.var()) == pytest.approx(1.0, abs=0.03)
    for h, expected in ((1, 0.9), (5, 0.5), (10, 0.0), (20, 0.0)):
        assert float(np.mean(w[:, :-h] * w[:, h:])) == pytest.approx(expected, abs=0.04)


def test_wave_rejects_smooth_below_one():
    with pytest.raises(ValueError, match="smooth"):
        K.wave(2, 10, 0, S.VRng(1))
    with pytest.raises(ValueError, match="smooth"):
        K.wave(2, 10, -3, S.VRng(1))
    assert K.wave(2, 10, 1, S.VRng(1)).shape == (2, 10)


def test_make_world_shapes_determinism_and_noise_free_values():
    a = K.make_world(8, 60, 0.5, 2.0, 7, 1.0, S.VRng(3))
    b = K.make_world(8, 60, 0.5, 2.0, 7, 1.0, S.VRng(3))
    assert all(v.shape == (8, 60) for v in a) and all(np.array_equal(x, y) for x, y in zip(a, b))
    X, Y, X0, Y0 = K.make_world(8, 60, 0.5, 0.0, 7, 0.5, S.VRng(3))
    assert np.array_equal(X, X0) and np.array_equal(Y, Y0)
    assert not np.array_equal(K.make_world(8, 60, 0.5, 2.0, 7, 1.0, S.VRng(4))[0], a[0])


def test_make_world_variances_follow_the_model():
    X, Y, _X0, _Y0 = K.make_world(300, 300, 0.5, 2.0, 7, 1.0, S.VRng(7))
    assert float(X.var()) == pytest.approx(2 + 4, rel=0.06) and float(Y.var()) == pytest.approx(1 + 1 + 4, rel=0.06)
    assert K.r_all(X, Y) == pytest.approx(K.exact_pooled_r(1, 7, 2.0, 1.0, 0.5), abs=0.04)


def test_make_world_validates_its_arguments():
    for args in ((8, 60, 1.0), (8, 60, -1.0)):
        with pytest.raises(ValueError):
            K.make_world(args[0], args[1], args[2], 1.0, 1, 1.0, S.VRng(1))
    with pytest.raises(ValueError):
        K.make_world(1, 60, 0.5, 1.0, 1, 1.0, S.VRng(1))
    with pytest.raises(ValueError):
        K.make_world(5, 1, 0.5, 1.0, 1, 1.0, S.VRng(1))


def test_window_means_by_hand():
    A = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]])
    assert K.window_means(A, 2).tolist() == [[1.5, 3.5], [15.0, 35.0]]               # der Rest (5 bzw. 50) wird verworfen
    assert K.window_means(A, 1).tolist() == A.tolist() and K.window_means(A, 5).tolist() == [[3.0], [30.0]]
    for bad in (0, 6):
        with pytest.raises(ValueError):
            K.window_means(A, bad)


# --- Zerlegung --------------------------------------------------------------------------------------------------------------------------

def test_split_levels_on_the_hand_instance(two_sites):
    X, Y = two_sites
    s = K.split_levels(X, Y)
    assert s["total"] == pytest.approx(96 / 104) and s["within"] == pytest.approx(-1.0) and s["between"] == pytest.approx(1.0)
    assert (s["ss_total"], s["ss_within"], s["ss_between"]) == pytest.approx((96.0, -4.0, 100.0))


def test_split_levels_identity_on_a_random_world():
    X, Y, _a, _b = K.make_world(12, 40, 0.5, 1.0, 7, -0.5, S.VRng(9))
    s = K.split_levels(X, Y)
    assert s["ss_total"] == pytest.approx(s["ss_within"] + s["ss_between"], rel=1e-12)


def test_r_all_flattens_and_matches_the_pairwise_value():
    A = np.array([[1.0, 2.0], [3.0, 5.0]])
    B = np.array([[2.0, 1.0], [4.0, 3.0]])
    assert K.r_all(A, B) == pytest.approx(K.pearson_r([1.0, 2.0, 3.0, 5.0], [2.0, 1.0, 4.0, 3.0]))


# --- Auswahl -------------------------------------------------------------------------------------------------------------------------------

def test_select_count_rounds_and_clamps():
    assert K.select_count(400, 0.05) == 20 and K.select_count(60, 0.2) == 12 and K.select_count(10, 0.05) == C.MIN_SELECTED and K.select_count(4, 1.0) == 4 and K.select_count(2, 0.5) == 2


def test_select_top_by_x_takes_the_largest_x():
    x = np.array([3.0, 9.0, 1.0, 7.0, 5.0])
    y = np.zeros(5)
    assert list(K.select_top(x, y, 0.6, "x")) == [1, 3, 4]


def test_select_top_by_the_standardized_sum():
    x = np.array([0.0, 10.0, 0.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    # x ist stärker gestreut als y: die standardisierte Summe stellt beide gleich, trotzdem liegt der Ausreißer in x vorn; der Gleichstand der übrigen entscheidet die stabile Reihenfolge
    idx = list(K.select_top(x, y, 0.6, "summe"))
    assert idx[0] in (1, 2) and set(idx) >= {1, 2}
    with pytest.raises(ValueError):
        K.select_top(x, y, 0.5, "nach_gefuehl")


def test_trunc_variance_ratio_hand_values():
    assert K.trunc_variance_ratio(1.0) == 1.0
    assert K.trunc_variance_ratio(0.5) == pytest.approx(1.0 - 2.0 / math.pi, abs=1e-12)
    assert K.trunc_variance_ratio(0.5) > K.trunc_variance_ratio(0.1) > K.trunc_variance_ratio(0.01) > 0
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="q muss"):
            K.trunc_variance_ratio(bad)


def test_exact_select_x_hand_values():
    """ρ = 0: bleibt 0; q = 1: bleibt ρ; ρ = 0,5 und obere Hälfte: u = 1 − 2/π = 0,3634, r = 0,5 √u / √(0,75 + 0,25 u) = 0,3287."""
    assert K.exact_select_x(0.0, 0.2) == 0.0 and K.exact_select_x(0.7, 1.0) == pytest.approx(0.7)
    assert K.exact_select_x(0.5, 0.5) == pytest.approx(0.3287, abs=0.0001)
    vals = [K.exact_select_x(0.5, q) for q in (1.0, 0.5, 0.2, 0.05, 0.01)]
    assert vals == sorted(vals, reverse=True) and vals[-1] > 0


def test_exact_select_sum_hand_values():
    """q = 1: ρ; ρ = 0,5 und obere 10 %: V_S = 3 · u(0,1) = 0,507, V_D = 1: r = (0,507 − 1)/(0,507 + 1) = −0,327."""
    assert K.exact_select_sum(0.4, 1.0) == pytest.approx(0.4)
    assert K.exact_select_sum(0.5, 0.1) == pytest.approx(-0.327, abs=0.001)
    assert K.exact_select_sum(0.5, 0.5) > 0 > K.exact_select_sum(0.5, 0.2)
    assert K.exact_select(0.5, 0.5, "x") == K.exact_select_x(0.5, 0.5) and K.exact_select(0.5, 0.5, "summe") == K.exact_select_sum(0.5, 0.5)


# --- Exakte Formeln der Welt ---------------------------------------------------------------------------------------------------------------

def test_wave_mean_variance_hand_values():
    assert K.wave_mean_variance(1, 30) == 1.0
    assert K.wave_mean_variance(4, 1) == pytest.approx(0.25)                                   # weißes Rauschen: 1/m
    assert K.wave_mean_variance(2, 30) == pytest.approx((2 + 2 * 1 * (1 - 1 / 30)) / 4)         # zwei Tage: (2 + 2 γ(1))/4
    assert K.wave_mean_variance(30, 30) > K.wave_mean_variance(90, 30) > 0


def test_exact_wave_window_r_is_constant_without_memory():
    """Welle ohne Gedächtnis (smooth = 1), γ = 1: V_m = 1/m, r = (1/m)/((1/m)(1 + σ²)) = 1/(1 + σ²) für jedes Fenster."""
    for m in (1, 7, 30, 90):
        assert K.exact_wave_window_r(m, 1, 2.0) == pytest.approx(0.2) and K.exact_wave_window_r(m, 1, 1.0) == pytest.approx(0.5)


def test_exact_wave_window_r_rises_with_the_window_when_the_wave_has_memory():
    vals = [K.exact_wave_window_r(m, 30, 2.0) for m in (1, 7, 30, 90)]
    assert vals == sorted(vals) and vals[0] == pytest.approx(0.2) and vals[1] == pytest.approx(0.618, abs=0.001) and vals[2] == pytest.approx(0.833, abs=0.001) and vals[-1] > 0.86
    assert K.exact_wave_window_r(7, 30, 2.0, -1.0) == pytest.approx(-K.exact_wave_window_r(7, 30, 2.0, 1.0))         # umgekehrte Kopplung: gleicher Betrag, anderes Vorzeichen


def test_exact_pooled_r_formula():
    """(ρ_b + γ V_m)/√(…): m = 1, ρ_b = 0,5, γ = 1, σ = 2: 1,5/6 = 0,25; mit σ = 0 und γ = 1 und ρ_b = 0: V/(V) = 1."""
    assert K.exact_pooled_r(1, 30, 2.0, 1.0, 0.5) == pytest.approx(0.25)
    assert K.exact_pooled_r(1, 7, 1.0, -0.5, 0.8) == pytest.approx(0.3 / math.sqrt(3.0 * 2.25))
    assert K.exact_pooled_r(30, 30, 0.0, 1.0, 0.0) == pytest.approx(1.0 * K.wave_mean_variance(30, 30) / (1 + K.wave_mean_variance(30, 30)), rel=1e-12)


def test_exact_within_window_r_for_one_day_windows_equals_the_within_formula():
    for smooth, sigma, gamma in ((1, 1.0, 1.0), (7, 2.0, 0.5), (30, 1.0, -0.5)):
        assert K.exact_within_window_r(120, 1, smooth, sigma, gamma) == pytest.approx(K.exact_within_r(120, smooth, sigma, gamma), abs=1e-12)


def test_exact_within_window_r_without_memory_is_nearly_constant():
    """Ohne Gedächtnis ist die Korrelation der Fenstermittel (ohne Niveau) fast für jedes Fenster 1/(1 + σ²) = 0,2; die Abweichung vom Standortmittel macht nur eine kleine Korrektur der Ordnung 1/T."""
    vals = [K.exact_within_window_r(360, m, 1, 2.0, 1.0) for m in (1, 7, 30, 90)]
    assert max(vals) - min(vals) < 0.02 and all(abs(v - 0.2) < 0.01 for v in vals)
    with pytest.raises(ValueError):
        K.exact_within_window_r(100, 101, 1, 1.0, 1.0)


def test_the_noise_free_correlation_is_the_attenuated_one_divided_by_the_reliabilities():
    for sigma, gamma, rho_b, smooth in ((2.0, 1.0, 0.5, 30), (1.0, -0.5, 0.8, 1), (0.5, 0.5, -0.3, 7)):
        rx, ry = K.reliabilities(sigma, gamma)
        assert K.attenuated_r(K.exact_true_r(smooth, gamma, rho_b), rx, ry) == pytest.approx(K.exact_pooled_r(1, smooth, sigma, gamma, rho_b), abs=1e-12)
        assert K.disattenuate(K.exact_pooled_r(1, smooth, sigma, gamma, rho_b), rx, ry) == pytest.approx(K.exact_true_r(smooth, gamma, rho_b), abs=1e-12)


def test_exact_within_r_hand_values():
    """σ = 0, γ = 1: innerhalb r = 1; γ = 0 ergibt 0; das Vorzeichen folgt γ."""
    assert K.exact_within_r(100, 7, 0.0, 1.0) == pytest.approx(1.0) and K.exact_within_r(100, 7, 1.0, 0.0) == 0.0
    assert K.exact_within_r(100, 7, 1.0, -0.5) < 0 < K.exact_within_r(100, 7, 1.0, 0.5)


def test_attenuation_and_correction_by_hand():
    assert K.attenuated_r(0.6, 0.5, 0.5) == pytest.approx(0.3) and K.attenuated_r(0.6, 1.0, 1.0) == 0.6
    assert K.disattenuate(0.3, 0.5, 0.5) == pytest.approx(0.6)
    for rx, ry in ((0.0, 0.5), (0.5, 0.0), (0.5, -1.0), (-1.0, 0.5)):
        with pytest.raises(ValueError, match="Zuverlässigkeit"):
            K.disattenuate(0.3, rx, ry)


def test_expected_total_r_hand_value():
    assert K.expected_total_r(0.8, -0.3) == pytest.approx(0.25) and K.expected_total_r(0.0, 0.5, 1.0, 1.0) == pytest.approx(0.25)
    assert K.expected_total_r(0.8, 0.0, 2.0, 1.0) == pytest.approx(3.2 / 5.0)
    assert K.expected_total_r(0.2, 0.6, 1.0, 2.0) == pytest.approx(2.6 / 5.0)                                  # Streuungen der Tage 2: Gewicht 4 gegen 1
    assert K.expected_total_r(0.5, 0.1, 3.0, 2.0) == pytest.approx((0.5 * 9 + 0.1 * 4) / 13.0)


def test_draw_pairs_and_noisy_pairs():
    x, y = K.draw_pairs(50_000, 0.6, S.VRng(3))
    assert K.pearson_r(x, y) == pytest.approx(0.6, abs=0.01)
    a, b = K.noisy_pairs(50_000, 0.6, 0.5, 0.8, S.VRng(4))
    assert float(a.var()) == pytest.approx(2.0, rel=0.03) and float(b.var()) == pytest.approx(1.25, rel=0.03) and K.pearson_r(a, b) == pytest.approx(K.attenuated_r(0.6, 0.5, 0.8), abs=0.01)


def test_standardize_gives_zero_mean_and_unit_spread():
    z = K.standardize([1.0, 2.0, 3.0])
    assert z.tolist() == pytest.approx([-math.sqrt(1.5), 0.0, math.sqrt(1.5)]) and float(z.mean()) == pytest.approx(0.0, abs=1e-15) and float(z.std()) == pytest.approx(1.0)
    assert K.standardize([10.0, 12.0, 14.0]).tolist() == pytest.approx(z.tolist())


def test_exact_within_window_r_rejects_a_window_of_zero_with_a_clear_message():
    for m in (0, -2):
        with pytest.raises(ValueError, match="Fenster"):
            K.exact_within_window_r(100, m, 30, 1.0, 1.0)
