"""Verdikte (reine Funktionen in agr_evaluation.py) an handgesetzten Werten: Art und jede Zahl im Text."""

import numpy as np

import agr_evaluation as ev


def _analysis(rows, **settings):
    s = ev.Settings(**{"g": 30, "t": 360, **settings})
    return ev.Analysis(settings=s, rows=rows, data={}, selected=None, split={}, rx=0.33, ry=0.33)


def _rows(total=0.27, between=0.59, within=0.19, window=None, selected=None, true=0.77, corrected=0.81):
    rows = [ev.Row("total", "Alle Standort-Tage", total, 0.25, 10800), ev.Row("true", "wahr", true, 0.75, 10800), ev.Row("corrected", "korrigiert", corrected, 0.75, 10800),
            ev.Row("within", "innerhalb", within, 0.187, 10800)]
    if window is not None:
        rows.append(ev.Row("window", "Fenster", window, 0.827, 360))
    rows.append(ev.Row("between", "zwischen", between, 0.53, 30))
    if selected is not None:
        rows.append(ev.Row("selected", "Auswahl", selected, 0.28, 6))
    return rows


def test_verdict_main_names_the_three_answers_and_the_expected_values():
    kind, text = ev.verdict_main(_analysis(_rows(between=0.45)))
    assert kind == "info"
    for needle in ("Dieselben 10 800 Werte", "alle Standort-Tage **0,27**", "Standortmittel **0,45**", "innerhalb der Standorte **0,19**", "Erwartet wären 0,25, 0,53 und 0,19"):
        assert needle in text, needle
    assert "Vorzeichen" not in text


def test_verdict_main_warns_for_different_signs_or_a_large_gap():
    kind, text = ev.verdict_main(_analysis(_rows(total=0.1, between=0.78, within=-0.32)))
    assert kind == "warning" and "Die Vorzeichen sind verschieden" in text and "-0,32" in text
    assert ev.verdict_main(_analysis(_rows(between=0.8, within=0.45)))[0] == "warning"          # Abstand 0,35 über 0,3
    assert ev.verdict_main(_analysis(_rows(between=0.7, within=0.45)))[0] == "info"             # Abstand 0,25 unter 0,3
    assert "Vorzeichen" not in ev.verdict_main(_analysis(_rows(between=0.7, within=-0.04)))[1]  # unter 0,05 zählt das Vorzeichen nicht


def test_verdict_window_only_with_a_window_and_names_points():
    assert ev.verdict_window(_analysis(_rows())) is None
    kind, text = ev.verdict_window(_analysis(_rows(window=0.83), window=30))
    assert kind == "info" and "Mittel über 30 Tage" in text and "von **0,19** auf **0,83**" in text and "erwartet 0,83" in text and "aus 10 800 Tagen werden 360 Fenster" in text
    assert "kein Gedächtnis" not in text
    assert "kein Gedächtnis" in ev.verdict_window(_analysis(_rows(window=0.2), window=30, smooth=1))[1]


def test_verdict_selection_only_with_a_selection_and_flags_sign_flips():
    assert ev.verdict_selection(_analysis(_rows())) is None
    kind, text = ev.verdict_selection(_analysis(_rows(selected=0.25), select="x", q=20))
    assert kind == "warning" and "obersten 20 % nach X (6 Standorte)" in text and "**0,25** statt 0,59 bei allen 30" in text and "erwartet 0,28" in text and "gekehrt" not in text
    kind, text = ev.verdict_selection(_analysis(_rows(selected=-0.2), select="summe", q=20))
    assert kind == "warning" and "nach der Summe X + Y" in text and "Die Auswahl hat das Vorzeichen gekehrt" in text
    assert ev.verdict_selection(_analysis(_rows(selected=0.45), select="x", q=50))[0] == "info"       # Abstand 0,14


def test_verdict_noise_only_with_noise():
    assert ev.verdict_noise(_analysis([r for r in _rows() if r.key != "corrected"])) is None
    kind, text = ev.verdict_noise(_analysis(_rows()))
    assert kind == "info" and "wäre r **0,77** statt 0,27" in text and "0,33 und 0,33" in text and "**0,81**" in text and "erwartet 0,75" in text


def _window_rows():
    def cell(smooth, sigma, m, points, mean, sd, exact, sig):
        return {"smooth": smooth, "sigma": sigma, "m": m, "points": points, "runs": 200, "mean": mean, "sd": sd, "exact": exact, "sig": sig}
    return [cell(30, 2.0, 1, 3600, 0.197, 0.023, 0.2, 1.0), cell(30, 2.0, 7, 514, 0.616, 0.035, 0.618, 1.0), cell(30, 2.0, 30, 120, 0.826, 0.031, 0.833, 1.0), cell(30, 2.0, 90, 40, 0.866, 0.041, 0.87, 1.0),
            cell(1, 2.0, 1, 3600, 0.201, 0.016, 0.2, 1.0), cell(1, 2.0, 7, 514, 0.198, 0.039, 0.2, 0.99), cell(1, 2.0, 30, 120, 0.206, 0.089, 0.2, 0.62), cell(1, 2.0, 90, 40, 0.196, 0.157, 0.2, 0.24)]


def test_verdict_window_grid_reads_first_middle_and_last_window():
    kind, text = ev.verdict_window_grid(_window_rows(), 30, 2.0)
    assert kind == "info"
    for needle in ("Glättung 30, Rauschen 2", "**0,197** bei Tagen", "**0,616** bei 7 Tagen", "**0,866** bei 90 Tagen", "exakt 0,200 / 0,618 / 0,870", "aus 3 600 Punkten werden 40", "von 0,023 auf 0,041"):
        assert needle in text, needle
    kind, text = ev.verdict_window_grid(_window_rows(), 1, 2.0)
    assert kind == "warning" and "von 100,0 % auf 24,0 %" in text and "von 0,016 auf 0,157" in text


def _select_rows():
    def cell(by, rho, q, k, mean, sd, exact, neg, sig):
        return {"by": by, "rho": rho, "q": q, "n": 400, "k": k, "runs": 2000, "mean": mean, "sd": sd, "neg": neg, "sig_pos": sig, "exact": exact}
    return [cell("x", 0.5, 1.0, 400, 0.5, 0.038, 0.5, 0.0, 1.0), cell("x", 0.5, 0.05, 20, 0.201, 0.224, 0.21, 0.188, 0.141),
            cell("summe", 0.5, 0.5, 10000, 0.046, 0.01, 0.043, 0.0, 1.0), cell("summe", 0.5, 0.01, 200, -0.545, 0.062, -0.55, 1.0, 0.0)]


def test_verdict_select_grid_reads_the_first_and_last_quota():
    kind, text = ev.verdict_select_grid(_select_rows(), "x", 0.5)
    assert kind == "warning" and "nach X" in text and "obersten 100,0 % der Einheiten zeigen **0,500** (exakt 0,500)" in text and "obersten 5,0 % **0,201** (exakt 0,210)" in text
    assert "Bei nur 20 gewählten Einheiten ist r in 18,8 % der Läufe negativ und nur in 14,1 % signifikant positiv" in text
    kind, text = ev.verdict_select_grid(_select_rows(), "summe", 0.5)
    assert "nach der Summe" in text and "**-0,545** (exakt -0,550)" in text and "negative Korrelation aus einer positiven" in text


def test_verdict_attenuation_grid_compares_the_worst_and_the_best_row():
    rows = [{"rx": 1.0, "ry": 1.0, "rho": 0.6, "exact": 0.6, "mean": 0.599, "sd": 0.037, "corrected_mean": 0.599, "corrected_sd": 0.037},
            {"rx": 0.4, "ry": 0.4, "rho": 0.6, "exact": 0.24, "mean": 0.237, "sd": 0.055, "corrected_mean": 0.593, "corrected_sd": 0.137}]
    kind, text = ev.verdict_attenuation_grid(rows)
    assert kind == "info"
    for needle in ("Wahres ρ = 0,6", "Zuverlässigkeit 0,4 und 0,4", "bleibt r **0,237** (exakt 0,240)", "**0,593**", "Streuung ist 0,137 statt 0,037"):
        assert needle in text, needle


def test_verdict_levels_grid_reads_one_row():
    rows = [{"g": 30, "t": 100, "rho_b": 0.8, "gamma": -0.5, "total": 0.101, "between": 0.777, "between_sd": 0.077, "within": -0.316, "exact_total": 0.115, "exact_between": 0.782, "exact_within": -0.316, "sign_differs": 1.0},
            {"g": 30, "t": 100, "rho_b": 0.8, "gamma": 0.5, "total": 0.495, "between": 0.789, "between_sd": 0.071, "within": 0.316, "exact_total": 0.5, "exact_between": 0.792, "exact_within": 0.316, "sign_differs": 0.0}]
    kind, text = ev.verdict_levels_grid(rows, 0)
    assert kind == "warning"
    for needle in ("30 Standorte mit je 100 Tagen, ρ_b = 0,8, Kopplung -0,5", "alle Standort-Tage **0,101**", "Standortmittel **0,777** (Streuung 0,077)", "innerhalb **-0,316**", "erwartet 0,115, 0,782, -0,316", "In 100,0 % der Läufe"):
        assert needle in text, needle
    assert "verschiedene Vorzeichen" not in ev.verdict_levels_grid(rows, 1)[1]


def test_helper_formats():
    assert ev._pct(0.1234) == "12,3 %" and ev._num(0.5, 2) == "0,50" and ev._n(0.5) == "0,5" and ev._n(2.0) == "2" and ev._int(10800) == "10 800"
    assert np.isfinite(0.0)
