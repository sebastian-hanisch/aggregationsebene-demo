"""Das vorgerechnete Raster (data/aggregation_grid.json): Aufbau, Plausibilität und Übereinstimmung mit dem Erzeuger (Probe)."""

import pytest

import agr_algorithm as K
import agr_constants as C
import agr_evaluation as ev


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None, "data/aggregation_grid.json fehlt: python tools/make_grid.py"
    return g


def _assert_close(a, b, path="probe"):
    """Verschachtelte Wörter und Listen: Zahlen mit relativer Toleranz (Summationsreihenfolge je Plattform, DEMO-PLAYBOOK §4), alles andere exakt."""
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert a == pytest.approx(b, rel=1e-8, abs=1e-10), path
    else:
        assert a == b, path


def test_probe_still_matches_the_generator(grid):
    """Ändert sich Erzeuger, Statistik oder Seed-Ableitung, passt die gespeicherte Probe nicht mehr: dann tools/make_grid.py neu laufen lassen."""
    _assert_close(ev.grid_probe(), grid["probe"])


def test_grid_has_every_cell_exactly_once(grid):
    def keys(rows, *names):
        return sorted(tuple(r[n] for n in names) for r in rows)

    assert keys(grid["window"], "smooth", "sigma", "m") == sorted((s, g, m) for s in C.WINDOW_SMOOTHS for g in C.WINDOW_SIGMAS for m in C.WINDOWS)
    assert keys(grid["select"], "by", "rho", "q") == sorted([("x", r, q) for r in C.SELECT_RHOS for q in C.SELECT_QS] + [("summe", r, q) for r in C.SELECT_RHOS for q in C.SUM_QS])
    assert keys(grid["attenuation"], "rx", "ry") == sorted(C.ATT_RELS) and len(grid["level"]) == len(C.LEVEL_CASES)


def test_grid_meta_and_run_counts(grid):
    assert grid["meta"]["seeds"] == list(C.GRID_SEEDS)
    assert all(r["runs"] == 4 * C.WINDOW_RUNS and r["points"] == C.WINDOW_T // r["m"] for r in grid["window"])
    assert all(r["runs"] == 4 * (C.SELECT_RUNS if r["by"] == "x" else C.SUM_RUNS) for r in grid["select"])
    assert all(r["runs"] == 4 * C.ATT_RUNS for r in grid["attenuation"]) and all(r["runs"] == 4 * C.LEVEL_RUNS for r in grid["level"])


def test_the_exact_columns_are_the_formulas(grid):
    for r in grid["window"]:
        assert r["exact"] == pytest.approx(K.exact_wave_window_r(r["m"], r["smooth"], r["sigma"]), rel=1e-12)
    for r in grid["select"]:
        assert r["exact"] == pytest.approx(K.exact_select(r["rho"], r["k"] / r["n"], r["by"]), rel=1e-12)
    for r in grid["attenuation"]:
        assert r["exact"] == pytest.approx(K.attenuated_r(r["rho"], r["rx"], r["ry"]), rel=1e-12)
    for r in grid["level"]:
        assert r["exact_total"] == pytest.approx(K.exact_pooled_r(1, r["smooth"], r["sigma"], r["gamma"], r["rho_b"]), rel=1e-12)
        assert r["exact_within"] == pytest.approx(K.exact_within_r(r["t"], r["smooth"], r["sigma"], r["gamma"]), rel=1e-12)


def test_measured_values_agree_with_the_exact_ones(grid):
    for r in grid["window"]:
        assert r["mean"] == pytest.approx(r["exact"], abs=3 * r["sd"] / r["runs"] ** 0.5 + 0.01), (r["smooth"], r["sigma"], r["m"])
    for r in grid["select"]:
        assert r["mean"] == pytest.approx(r["exact"], abs=3 * r["sd"] / r["runs"] ** 0.5 + 0.02), (r["by"], r["rho"], r["q"])
    for r in grid["attenuation"]:
        assert r["mean"] == pytest.approx(r["exact"], abs=0.01) and r["corrected_mean"] == pytest.approx(r["rho"], abs=0.02)
    for r in grid["level"]:
        assert r["total"] == pytest.approx(r["exact_total"], abs=0.04 if r["g"] >= 30 else 0.06) and r["between"] == pytest.approx(r["exact_between"], abs=0.03) and r["within"] == pytest.approx(r["exact_within"], abs=0.01)


def test_without_memory_the_window_does_not_change_r_but_costs_points(grid):
    cells = sorted((r for r in grid["window"] if r["smooth"] == 1 and r["sigma"] == 2.0), key=lambda r: r["m"])
    assert max(r["mean"] for r in cells) - min(r["mean"] for r in cells) < 0.03
    assert [r["sd"] for r in cells] == sorted(r["sd"] for r in cells) and cells[-1]["sd"] > 5 * cells[0]["sd"] and cells[-1]["sig"] < 0.35 and cells[0]["sig"] == 1.0


def test_with_memory_the_window_raises_r(grid):
    cells = sorted((r for r in grid["window"] if r["smooth"] == 30 and r["sigma"] == 2.0), key=lambda r: r["m"])
    assert [r["mean"] for r in cells] == sorted(r["mean"] for r in cells) and cells[0]["mean"] < 0.25 and cells[-1]["mean"] > 0.8


def test_selection_by_x_weakens_r_with_the_quota_and_few_units_make_it_uncertain(grid):
    for rho in C.SELECT_RHOS:
        cells = sorted((r for r in grid["select"] if r["by"] == "x" and r["rho"] == rho), key=lambda r: -r["q"])
        assert [r["mean"] for r in cells] == sorted((r["mean"] for r in cells), reverse=True) and cells[0]["mean"] == pytest.approx(rho, abs=0.02)
        assert cells[-1]["sd"] > 3 * cells[0]["sd"]
    small = [r for r in grid["select"] if r["by"] == "x" and r["rho"] == 0.5 and r["q"] == 0.05][0]
    assert small["k"] == 20 and 0.1 < small["neg"] < 0.3 and small["sig_pos"] < 0.2


def test_selection_by_the_sum_turns_positive_into_negative(grid):
    half = {r["rho"]: r for r in grid["select"] if r["by"] == "summe" and r["q"] == 0.5}
    assert half[0.3]["mean"] < 0 < half[0.5]["mean"] < half[0.8]["mean"]
    tenth = [r for r in grid["select"] if r["by"] == "summe" and r["rho"] == 0.5 and r["q"] == 0.1][0]
    assert tenth["mean"] < -0.3 and tenth["neg"] == 1.0


def test_the_correction_is_unbiased_but_scatters_more(grid):
    rows = sorted(grid["attenuation"], key=lambda r: -(r["rx"] * r["ry"]))
    assert [r["corrected_sd"] for r in rows] == sorted(r["corrected_sd"] for r in rows) and rows[-1]["corrected_sd"] > 3 * rows[0]["corrected_sd"]
    assert [r["mean"] for r in rows] == sorted((r["mean"] for r in rows), reverse=True)


def test_one_world_gives_three_answers(grid):
    first = grid["level"][0]
    assert first["within"] < 0 < first["between"] and first["sign_differs"] > 0.99 and first["total"] < first["between"] - 0.5
    assert grid["level"][1]["between_sd"] > 1.5 * first["between_sd"]                        # weniger Standorte, mehr Streuung
