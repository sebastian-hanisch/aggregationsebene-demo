"""Auswertung: Einzelwelt, Zellen des Rasters (klein gerechnet), Zugriff auf das Raster."""

import numpy as np
import pytest

import agr_algorithm as K
import agr_constants as C
import agr_evaluation as ev
import prs_scenario as S


def test_settings_equality_and_hash_use_the_values():
    assert ev.Settings() == ev.Settings() and hash(ev.Settings()) == hash(ev.Settings())
    assert ev.Settings(rho_b=0.9) != ev.Settings() and ev.Settings() != "x"
    assert len({ev.Settings(), ev.Settings(), ev.Settings(seed=1)}) == 2


def test_analyse_is_deterministic_and_consistent_with_the_kernel():
    s = ev.Settings(12, 120, 0.5, 1.0, 7, 0.5, 7, "x", 50, 7)
    a, b = ev.analyse(s), ev.analyse(s)
    assert [r.r for r in a.rows] == [r.r for r in b.rows] and a.split == b.split
    data_seed = S.substream_seeds(7, 2)[0]
    X, Y, _x0, _y0 = K.make_world(12, 120, 0.5, 1.0, 7, 0.5, S.VRng(data_seed))
    sp = K.split_levels(X, Y)
    by = {r.key: r for r in a.rows}
    assert by["total"].r == sp["total"] and by["between"].r == sp["between"] and by["within"].r == sp["within"]
    assert by["total"].exact == K.exact_pooled_r(1, 7, 1.0, 0.5, 0.5) and by["between"].exact == K.exact_pooled_r(120, 7, 1.0, 0.5, 0.5)
    assert by["within"].exact == K.exact_within_r(120, 7, 1.0, 0.5) and by["true"].exact == K.exact_true_r(7, 0.5, 0.5)
    assert by["total"].points == 12 * 120 and by["between"].points == 12 and by["window"].points == 12 * (120 // 7)


def test_the_rows_depend_on_the_settings():
    keys = lambda **kw: [r.key for r in ev.analyse(ev.Settings(**kw)).rows]       # noqa: E731
    assert keys() == ["total", "true", "corrected", "within", "between"]
    assert keys(window=30)[-2:] == ["window", "between"] and "selected" not in keys()
    assert keys(select="x")[-1] == "selected" and keys(select="summe")[-1] == "selected"
    assert "corrected" not in keys(sigma=0.0)


def test_the_selection_row_uses_the_chosen_standorte():
    a = ev.analyse(ev.Settings(g=40, select="x", q=25))
    sel = ev.row(a, "selected")
    assert len(a.selected) == 10 and sel.points == 10
    mx = a.data["between"][0]
    assert set(a.selected) == set(np.argsort(-mx)[:10])
    assert ev.row(ev.analyse(ev.Settings()), "selected") is None and ev.analyse(ev.Settings()).selected is None


def test_the_corrected_row_is_the_total_divided_by_the_reliabilities():
    a = ev.analyse(ev.Settings(sigma=2.0, gamma=1.0))
    rx, ry = K.reliabilities(2.0, 1.0)
    assert (a.rx, a.ry) == (rx, ry) and ev.row(a, "corrected").r == pytest.approx(ev.row(a, "total").r / np.sqrt(rx * ry), abs=1e-12)
    assert ev.row(a, "corrected").exact == pytest.approx(ev.row(a, "true").exact)


def test_cell_seed_differs_for_every_input():
    base = ev.cell_seed("window", 7, 2.0, 30, 101)
    others = {ev.cell_seed("select", 7, 2.0, 30, 101), ev.cell_seed("window", 30, 2.0, 30, 101), ev.cell_seed("window", 7, 1.0, 30, 101), ev.cell_seed("window", 7, 2.0, 7, 101), ev.cell_seed("window", 7, 2.0, 30, 102)}
    assert base not in others and len(others) == 5


def test_window_cell_small_run():
    cell = ev.window_cell(7, 1.0, 7, 3, (7,), 600)
    assert cell["points"] == 600 // 7 and cell["runs"] == 3 and cell["exact"] == K.exact_wave_window_r(7, 7, 1.0) and 0.5 < cell["mean"] < 1.0
    assert cell == ev.window_cell(7, 1.0, 7, 3, (7,), 600)


def test_select_cell_small_runs():
    x = ev.select_cell("x", 0.5, 0.2, 100, 6, (7,))
    assert x["k"] == 20 and x["runs"] == 6 and x["exact"] == K.exact_select_x(0.5, 0.2) and 0.0 <= x["neg"] <= 1.0
    s = ev.select_cell("summe", 0.5, 0.1, 2000, 2, (7,))
    assert s["k"] == 200 and s["exact"] == K.exact_select_sum(0.5, 0.1) and s["mean"] < 0 and s["neg"] == 1.0
    assert x != ev.select_cell("x", 0.5, 0.2, 100, 6, (8,))


def test_attenuation_cell_small_run():
    cell = ev.attenuation_cell(0.6, 0.8, 4, (7,), 400)
    assert cell["exact"] == pytest.approx(0.6 * np.sqrt(0.48)) and cell["runs"] == 4 and cell["corrected_mean"] == pytest.approx(cell["mean"] / np.sqrt(0.48), abs=1e-12)
    assert ev.attenuation_cell(1.0, 1.0, 2, (7,), 200)["exact"] == 0.6


def test_level_cell_small_run():
    cell = ev.level_cell(10, 40, 0.8, 1.0, 1, -0.5, 4, (7,))
    assert cell["runs"] == 4 and cell["exact_within"] == K.exact_within_r(40, 1, 1.0, -0.5) and cell["exact_between"] == K.exact_pooled_r(40, 1, 1.0, -0.5, 0.8)
    assert cell["within"] < 0 < cell["between"] and cell["sign_differs"] == 1.0


def test_find_matches_floats_and_returns_none_when_missing():
    rows = [{"rho": 0.5, "by": "x"}, {"rho": 0.8, "by": "summe"}]
    assert ev.find(rows, rho=0.8, by="summe") == rows[1] and ev.find(rows, rho=0.3) is None and ev.find(rows, rho=0.5 + 1e-12) == rows[0]


def test_load_grid_returns_none_for_a_missing_or_broken_file(tmp_path):
    assert ev.load_grid(tmp_path / "gibt-es-nicht.json") is None
    bad = tmp_path / "kaputt.json"
    bad.write_text("{", encoding="utf-8")
    assert ev.load_grid(bad) is None


def test_the_grid_constants_cover_the_cells():
    assert set(C.WINDOW_SMOOTHS) <= set(C.SMOOTHS) and set(C.WINDOWS) >= {1, 7, 30, 90} and all(len(case) == 6 for case in C.LEVEL_CASES)
