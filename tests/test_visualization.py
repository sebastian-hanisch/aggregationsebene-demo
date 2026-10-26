"""Abbildungen: jede wird gebaut, hat die erwarteten Spuren und Achsen."""

import numpy as np
import pytest

import agr_constants as C
import agr_evaluation as ev
import agr_visualization as V


@pytest.fixture(scope="module")
def grid():
    return ev.load_grid()


def test_every_chart_builds_and_locks_its_axes(grid):
    a = ev.analyse(ev.Settings(12, 120, 0.5, 1.0, 7, 0.5, 7, "x", 50, 3))
    x, y = a.data["between"]
    figs = [V.build_scatter(x, y, 0.5, "Standortmittel", C.COLOR_BETWEEN, selected=a.selected), V.build_rows_bar(a.rows), V.build_window(grid["window"], 2.0), V.build_select(grid["select"], "x"),
            V.build_select(grid["select"], "summe"), V.build_attenuation(grid["attenuation"]), V.build_levels(grid["level"])]
    for fig in figs:
        assert fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange and len(fig.data) >= 1


def test_scatter_highlights_the_selected_sites_and_thins_large_clouds():
    x = np.arange(10.0)
    y = x ** 0.5
    fig = V.build_scatter(x, y, 0.9, "Test", C.COLOR_ALL, selected=[7, 8, 9])
    assert [t.name for t in fig.data][:2] == ["alle", "gewählte Standorte"] and list(fig.data[1].x) == [7.0, 8.0, 9.0] and len(fig.data[0].x) == 7
    assert fig.layout.title.text == "Test" and fig.data[-1].name.startswith("Gerade (r = 0,900")
    plain = V.build_scatter(x, y, 0.9, "Test", C.COLOR_ALL)
    assert [t.name for t in plain.data][0] == "Werte" and len(plain.data[0].x) == 10
    big = np.arange(10_000.0)
    thin = V.build_scatter(big, big, 1.0, "gross", C.COLOR_ALL, max_points=1000)
    assert len(thin.data[0].x) == 1000


def test_scatter_without_variation_has_no_line():
    fig = V.build_scatter(np.ones(5), np.arange(5.0), 0.0, "konstant", C.COLOR_ALL)
    assert [t.name for t in fig.data] == ["Werte"]


def test_rows_bar_shows_measured_values_and_expected_diamonds():
    rows = [ev.Row("total", "A", 0.27, 0.25, 100), ev.Row("within", "B", 0.19, None, 50)]
    fig = V.build_rows_bar(rows)
    assert [t.name for t in fig.data] == ["gemessen", "erwartet (exakt)"] and list(fig.data[0].x) == [0.27, 0.19] and list(fig.data[1].y) == ["Alle Standort-Tage"] and list(fig.data[0].y) == ["Alle Standort-Tage", "Innerhalb der Standorte"] and list(fig.data[0].customdata) == ["A", "B"] and list(fig.data[1].x) == [0.25]
    assert fig.layout.yaxis.autorange == "reversed"


def test_window_chart_has_a_line_and_diamonds_per_memory(grid):
    fig = V.build_window(grid["window"], 2.0)
    assert [t.name for t in fig.data if t.showlegend is not False] == ["kein Gedächtnis", "Gedächtnis 7 Tage", "Gedächtnis 30 Tage"] and len(fig.data) == 6
    assert list(fig.data[0].x) == ["1", "7", "30", "90"]
    cell = ev.find(grid["window"], smooth=30, sigma=2.0, m=30)
    assert fig.data[4].y[2] == pytest.approx(cell["mean"]) and fig.data[5].y[2] == pytest.approx(cell["exact"])


def test_select_chart_has_one_line_per_rho_and_the_zero_line(grid):
    fig = V.build_select(grid["select"], "summe")
    assert [t.name for t in fig.data if t.showlegend is not False] == ["ρ = 0,3", "ρ = 0,5", "ρ = 0,8"] and list(fig.data[0].x) == ["50 %", "20 %", "10 %", "1 %"]
    assert any(s.y0 == 0 for s in fig.layout.shapes) and list(fig.layout.yaxis.range) == [-1, 1]
    assert list(V.build_select(grid["select"], "x").data[0].x) == ["100 %", "50 %", "20 %", "5 %"]


def test_attenuation_chart_orders_by_reliability_and_marks_the_true_rho(grid):
    fig = V.build_attenuation(grid["attenuation"])
    assert list(fig.data[0].x) == ["1 / 1", "0,8 / 0,8", "0,5 / 0,8", "0,6 / 0,6", "0,4 / 0,4"]
    assert [t.name for t in fig.data] == ["gemessen", "korrigiert", "exakt verdünnt"] and fig.layout.barmode == "group"
    assert any(s.y0 == 0.6 for s in fig.layout.shapes)


def test_levels_chart_has_three_bars_and_diamonds_per_world(grid):
    fig = V.build_levels(grid["level"])
    assert [t.name for t in fig.data if t.showlegend is not False] == ["alle Standort-Tage", "Standortmittel", "innerhalb der Standorte"]
    assert len(fig.data[0].x) == len(C.LEVEL_CASES) and fig.data[0].x[0].startswith("30 Standorte, ρ_b 0,8, Kopplung -0,5")
