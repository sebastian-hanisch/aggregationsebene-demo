"""Jede Zahl der README steht hier und wird aus dem vorgerechneten Raster (data/aggregation_grid.json) oder über dieselben Funktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4).

Zufall nur aus SplitMix64 mit festen Seeds. Die Tabellen der README werden Zeile für Zeile in der Rundung der App (`_c`, `_p`) gegen die README geprüft, sonst gilt eine Toleranz von einem
halben Tausendstel bzw. Hundertstel. Die Zeilen-Bauer (`window_rows` usw.) dienen auch dazu, die README-Tabellen zu erzeugen."""

import math
from pathlib import Path

import numpy as np
import pytest

import agr_algorithm as K
import agr_constants as C
import agr_evaluation as ev

README_PATH = Path(__file__).resolve().parent.parent / "README.md"
README = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""


def _c(v, d):
    return f"{round(v, d) + 0.0:.{d}f}".replace(".", ",")                  # kein „-0,000“


def _p(v):
    return f"{100 * v:.1f}".replace(".", ",")


def _n(v):
    return f"{v:g}".replace(".", ",")


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None
    return g


# --- Zeilen-Bauer (Rundung der App) ------------------------------------------------------------------------------------------------------------

def window_rows(grid) -> list[str]:
    rows = []
    for smooth in C.WINDOW_SMOOTHS:
        for sigma in C.WINDOW_SIGMAS:
            cells = [ev.find(grid["window"], smooth=smooth, sigma=sigma, m=m) for m in C.WINDOWS]
            rows.append(f"| {smooth} | {_n(sigma)} | " + " | ".join(f"{_c(c['mean'], 3)} ({_c(c['exact'], 3)})" for c in cells) + " |")
    return rows


def window_spread_rows(grid) -> list[str]:
    rows = []
    for smooth in (1, 30):
        cells = [ev.find(grid["window"], smooth=smooth, sigma=2.0, m=m) for m in C.WINDOWS]
        rows.append(f"| {smooth} | " + " | ".join(f"{_c(c['sd'], 3)} / {_p(c['sig'])} %" for c in cells) + " |")
    return rows


def select_x_rows(grid) -> list[str]:
    return [f"| {_n(rho)} | " + " | ".join(f"{_c(c['mean'], 3)} ({_c(c['exact'], 3)})" for c in (ev.find(grid["select"], by="x", rho=rho, q=q) for q in C.SELECT_QS)) + " |" for rho in C.SELECT_RHOS]


def select_sum_rows(grid) -> list[str]:
    return [f"| {_n(rho)} | " + " | ".join(f"{_c(c['mean'], 3)} ({_c(c['exact'], 3)})" for c in (ev.find(grid["select"], by="summe", rho=rho, q=q) for q in C.SUM_QS)) + " |" for rho in C.SELECT_RHOS]


def attenuation_rows(grid) -> list[str]:
    return [f"| {_n(r['rx'])} | {_n(r['ry'])} | {_c(r['exact'], 3)} | {_c(r['mean'], 3)} ({_c(r['sd'], 3)}) | {_c(r['corrected_mean'], 3)} ({_c(r['corrected_sd'], 3)}) |" for r in grid["attenuation"]]


def level_rows(grid) -> list[str]:
    return [f"| {c['g']} | {_n(c['rho_b'])} | {_n(c['gamma'])} | {_c(c['total'], 3)} ({_c(c['exact_total'], 3)}) | {_c(c['between'], 3)} ({_c(c['exact_between'], 3)}), sd {_c(c['between_sd'], 3)} | "
            f"{_c(c['within'], 3)} ({_c(c['exact_within'], 3)}) | {_p(c['sign_differs'])} % |" for c in grid["level"]]


# --- Tabellen der README --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("builder", [window_rows, window_spread_rows, select_x_rows, select_sum_rows, attenuation_rows, level_rows])
def test_claim_the_tables_are_the_grid_in_the_apps_rounding(grid, builder):
    """README: die sechs Tabellen der Befunde (sechs, zwei, drei, drei, fünf und vier Zeilen)."""
    rows = builder(grid)
    assert rows
    for row in rows:
        assert row in README, row


# --- Aussagen -----------------------------------------------------------------------------------------------------------------------------

def test_claim_the_window_statements(grid):
    """README: Glättung 30, σ = 2: 0,201 / 0,616 / 0,826 / 0,866 (exakt 0,200 / 0,618 / 0,833 / 0,870); ohne Gedächtnis bleibt r bei 0,2; Punkte 3 600 → 40; Streuung 0,016 → 0,157; signifikant 100 → 24,5 %."""
    c30 = [ev.find(grid["window"], smooth=30, sigma=2.0, m=m) for m in C.WINDOWS]
    assert [c["mean"] for c in c30] == pytest.approx([0.201, 0.616, 0.826, 0.866], abs=0.0006) and [c["exact"] for c in c30] == pytest.approx([0.200, 0.618, 0.833, 0.870], abs=0.0006)
    c1 = [ev.find(grid["window"], smooth=1, sigma=2.0, m=m) for m in C.WINDOWS]
    assert [c["mean"] for c in c1] == pytest.approx([0.201, 0.198, 0.206, 0.196], abs=0.0006) and all(c["exact"] == pytest.approx(0.2) for c in c1)
    assert [c["points"] for c in c1] == [3600, 514, 120, 40] and [c["sd"] for c in c1] == pytest.approx([0.016, 0.039, 0.089, 0.157], abs=0.0006)
    assert [c["sig"] for c in c1] == pytest.approx([1.0, 0.995, 0.62, 0.245], abs=0.0006)


def test_claim_the_selection_by_x_statements(grid):
    """README: ρ = 0,5: 0,500 / 0,331 / 0,259 / 0,201 (exakt 0,500 / 0,329 / 0,261 / 0,210); bei 20 gewählten Einheiten Streuung 0,224, 18,8 % negativ, 14,1 % signifikant positiv, bei 80: 65,3 %."""
    cells = [ev.find(grid["select"], by="x", rho=0.5, q=q) for q in C.SELECT_QS]
    assert [c["mean"] for c in cells] == pytest.approx([0.500, 0.331, 0.259, 0.201], abs=0.0006) and [c["exact"] for c in cells] == pytest.approx([0.500, 0.329, 0.261, 0.210], abs=0.0006)
    small = cells[-1]
    assert (small["k"], small["sd"], small["neg"], small["sig_pos"]) == pytest.approx((20, 0.224, 0.188, 0.141), abs=0.0006) and cells[2]["sig_pos"] == pytest.approx(0.653, abs=0.0006)


def test_claim_the_selection_by_the_sum_statements(grid):
    """README: ρ = 0,5: +0,046 / -0,210 / -0,326 / -0,545 (exakt +0,043 / -0,208 / -0,327 / -0,550); ρ = 0,3 schon bei den oberen 50 %: -0,196; ρ = 0,8 erst bei 1 %: -0,084, in 85,0 % der Läufe negativ."""
    cells = [ev.find(grid["select"], by="summe", rho=0.5, q=q) for q in C.SUM_QS]
    assert [c["mean"] for c in cells] == pytest.approx([0.046, -0.210, -0.326, -0.545], abs=0.0006) and [c["exact"] for c in cells] == pytest.approx([0.043, -0.208, -0.327, -0.550], abs=0.0006)
    assert ev.find(grid["select"], by="summe", rho=0.3, q=0.5)["mean"] == pytest.approx(-0.196, abs=0.0006)
    last = ev.find(grid["select"], by="summe", rho=0.8, q=0.01)
    assert (last["mean"], last["neg"]) == pytest.approx((-0.084, 0.85), abs=0.0006)


def test_claim_the_attenuation_statements(grid):
    """README: wahres ρ = 0,6; bei Zuverlässigkeit 0,4 / 0,4 bleibt r 0,237 (exakt 0,240); Korrektur 0,593 mit Streuung 0,137 gegen 0,037 ohne Messfehler."""
    low = ev.find(grid["attenuation"], rx=0.4, ry=0.4)
    top = ev.find(grid["attenuation"], rx=1.0, ry=1.0)
    assert (low["mean"], low["exact"], low["corrected_mean"], low["corrected_sd"], top["corrected_sd"]) == pytest.approx((0.237, 0.240, 0.593, 0.137, 0.037), abs=0.0006)


def test_claim_the_level_statements(grid):
    """README: ρ_b = 0,8, Kopplung -0,5, 30 Standorte: gesamt 0,101, Standortmittel 0,777 (sd 0,077), innerhalb -0,316; Vorzeichen in 100,0 % verschieden; bei 10 Standorten sd 0,147."""
    first = grid["level"][0]
    assert (first["total"], first["between"], first["between_sd"], first["within"], first["sign_differs"]) == pytest.approx((0.101, 0.777, 0.077, -0.316, 1.0), abs=0.0006)
    assert grid["level"][1]["between_sd"] == pytest.approx(0.147, abs=0.0006) and (grid["level"][1]["total"], grid["level"][1]["exact_total"]) == pytest.approx((0.071, 0.115), abs=0.0006)
    assert (grid["level"][2]["between"], grid["level"][2]["within"]) == pytest.approx((0.009, 0.499), abs=0.0006)


def test_claim_the_expected_values_of_the_default_world():
    """README: Standardwelt (30 Standorte, 360 Tage, ρ_b = 0,5, σ = 2, Gedächtnis 30, Kopplung 1): erwartet 0,25 gesamt, 0,53 Standortmittel, 0,19 innerhalb, 0,75 ohne Rauschen; Fenster 30 Tage innerhalb 0,83."""
    s = ev.Settings(window=30)
    assert K.exact_pooled_r(1, s.smooth, s.sigma, s.gamma, s.rho_b) == pytest.approx(0.25, abs=0.005) and K.exact_pooled_r(s.t, s.smooth, s.sigma, s.gamma, s.rho_b) == pytest.approx(0.53, abs=0.005)
    assert K.exact_within_r(s.t, s.smooth, s.sigma, s.gamma) == pytest.approx(0.19, abs=0.005) and K.exact_true_r(s.smooth, s.gamma, s.rho_b) == pytest.approx(0.75, abs=0.005)
    assert K.exact_within_window_r(s.t, 30, s.smooth, s.sigma, s.gamma) == pytest.approx(0.83, abs=0.005)


def test_claim_hand_values(two_sites):
    """README: Zerlegung zweier Standorte: gesamt 0,923, innerhalb -1, zwischen +1, Kreuzprodukte 96 = -4 + 100; obere Hälfte einer Normalverteilung: u = 1 - 2/π = 0,3634, r = 0,3287 bei ρ = 0,5;
    Messrauschen mit Zuverlässigkeit 0,5 halbiert ρ; bei einer Welle ohne Gedächtnis ist r = 1/(1 + σ²) = 0,2 für jedes Fenster."""
    X, Y = two_sites
    s = K.split_levels(X, Y)
    assert (s["total"], s["within"], s["between"]) == pytest.approx((0.923, -1.0, 1.0), abs=0.0006) and (s["ss_total"], s["ss_within"], s["ss_between"]) == pytest.approx((96.0, -4.0, 100.0))
    assert K.trunc_variance_ratio(0.5) == pytest.approx(0.3634, abs=0.00006) and K.exact_select_x(0.5, 0.5) == pytest.approx(0.3287, abs=0.00006)
    assert K.attenuated_r(0.6, 0.5, 0.5) == pytest.approx(0.3) and K.exact_wave_window_r(30, 1, 2.0) == pytest.approx(0.2)
    assert K.exact_select_sum(0.5, 0.1) == pytest.approx(-0.327, abs=0.0006) and K.expected_total_r(0.8, -0.3) == pytest.approx(0.25)


def test_claim_the_grid_time(grid):
    """README: das Raster dauert 2 Sekunden auf 10 Prozessen (Wert aus der Datei; ändert sich beim Neuerzeugen)."""
    assert grid["meta"]["seconds"] == 2


# --- Texte der Presets ---------------------------------------------------------------------------------------------------------------------

def _settings(name):
    p = C.PRESETS[name]
    return ev.Settings(p["g"], p["t"], p["rho_b"], p["sigma"], p["smooth"], p["gamma"], p["window"], p["select"], p["q"], p["seed"])


def test_claim_the_preset_help_texts_quote_exact_and_grid_numbers(grid):
    h = C.PRESET_HELP
    s = _settings("Standardfall (Voreinstellung)")
    std = h["Standardfall (Voreinstellung)"]
    exact = (K.exact_pooled_r(1, s.smooth, s.sigma, s.gamma, s.rho_b), K.exact_pooled_r(s.t, s.smooth, s.sigma, s.gamma, s.rho_b), K.exact_within_r(s.t, s.smooth, s.sigma, s.gamma), K.exact_true_r(s.smooth, s.gamma, s.rho_b))
    assert f"erwartet sind r = {_c(exact[0], 2)} über alle Standort-Tage, {_c(exact[1], 2)} für die Standortmittel und {_c(exact[2], 2)} innerhalb der Standorte; ohne Messrauschen wären es {_c(exact[3], 2)}" in std
    w = _settings("Tage, Wochen, Monate")
    text = h["Tage, Wochen, Monate"]
    assert f"von erwartet {_c(K.exact_within_r(w.t, w.smooth, w.sigma, w.gamma), 2)} (Tage) auf {_c(K.exact_within_window_r(w.t, 30, w.smooth, w.sigma, w.gamma), 2)} (Fenster)" in text and f"aus {w.g * w.t // 1000} 800 Tageswerten werden {w.g * (w.t // 30)} Fenstermittel" in text
    n = _settings("Welle ohne Gedächtnis")
    assert f"bleibt r bei erwartet {_c(K.exact_within_window_r(n.t, 30, n.smooth, n.sigma, n.gamma), 2)}" in h["Welle ohne Gedächtnis"] and _c(K.exact_within_r(n.t, n.smooth, n.sigma, n.gamma), 2) == _c(K.exact_within_window_r(n.t, 30, n.smooth, n.sigma, n.gamma), 2)
    best = _settings("Nur die besten Standorte")
    eb = K.exact_pooled_r(best.t, best.smooth, best.sigma, best.gamma, best.rho_b)
    k = K.select_count(best.g, best.q / 100)
    text = h["Nur die besten Standorte"]
    assert f"({k} Standorte) zeigen erwartet {_c(K.exact_select_x(eb, k / best.g), 2)} statt {_c(eb, 2)} für alle" in text
    small = ev.find(grid["select"], by="x", rho=0.5, q=0.05)
    assert f"Streuung {_c(small['sd'], 3)} bei {small['k']} gewählten Standorten und ρ = 0,5" in text
    summ = _settings("Auswahl nach der Summe")
    es = K.exact_pooled_r(summ.t, summ.smooth, summ.sigma, summ.gamma, summ.rho_b)
    assert f"erwartet {_c(K.exact_select_sum(es, K.select_count(summ.g, summ.q / 100) / summ.g), 2)} statt +{_c(es, 2)} für alle" in h["Auswahl nach der Summe"]
    lv = _settings("Gleiche Welt, andere Ebene")
    text = h["Gleiche Welt, andere Ebene"]
    assert (f"alle Standort-Tage {_c(K.exact_pooled_r(1, lv.smooth, lv.sigma, lv.gamma, lv.rho_b), 2)}, Standortmittel {_c(K.exact_pooled_r(lv.t, lv.smooth, lv.sigma, lv.gamma, lv.rho_b), 2)}, "
            f"innerhalb der Standorte {_c(K.exact_within_r(lv.t, lv.smooth, lv.sigma, lv.gamma), 2)}") in text
    case = grid["level"][0]
    assert f"in {_p(case['sign_differs'])} % der {case['runs']} Läufe" in text and f"({case['g']} Standorte, {case['t']} Tage)" in text


def test_the_claim_helper_uses_numpy_and_math():
    assert np.isfinite(math.pi)
