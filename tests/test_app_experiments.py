"""App: jedes Preset, der Permalink, jedes Experiment-Auswahlfeld einzeln und die Eckfall-Kombinationen (DEMO-PLAYBOOK §1d)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import agr_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([m.value for m in at.markdown] + [s.value for s in at.success] + [i.value for i in at.info] + [w.value for w in at.warning])


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_shows_its_data(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    preset = C.PRESETS[name]
    assert at.session_state["g_slider"] == preset["g"] and at.session_state["rho_b_slider"] == preset["rho_b"] and at.session_state["smooth_select"] == preset["smooth"]
    assert at.session_state["window_select"] == preset["window"] and at.session_state["select_select"] == preset["select"] and at.session_state["gamma_slider"] == preset["gamma"]
    assert len(at.get("plotly_chart")) == 2 + 4                  # Hauptansicht: Balken und Streudiagramm; E1 bis E4: je eine Abbildung


def test_permalink_is_loaded_and_clamped():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["g"] = "9999"
    at.query_params["t"] = "1"
    at.query_params["smooth"] = "7"
    at.query_params["select"] = "summe"
    at.query_params["window"] = "abc"                # ungültig: wird ignoriert
    at.query_params["gamma"] = "-9"
    at.run()
    assert not at.exception
    assert at.session_state["g_slider"] == C.G_MAX and at.session_state["t_slider"] == C.T_MIN and at.session_state["smooth_select"] == 7
    assert at.session_state["select_select"] == "summe" and at.session_state["window_select"] == C.DEFAULT_WINDOW and at.session_state["gamma_slider"] == C.GAMMA_MIN


def test_permalink_ignores_unknown_choices():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["smooth"] = "8"
    at.query_params["select"] = "zufall"
    at.run()
    assert not at.exception and at.session_state["smooth_select"] == C.DEFAULT_SMOOTH and at.session_state["select_select"] == "keine"


def test_six_metrics_show_the_cuts():
    at = _app()
    assert [m.label for m in at.metric] == ["Alle Standort-Tage", "Standortmittel (zwischen)", "Innerhalb der Standorte", "Fenster", "Auswahl", "Ohne Messrauschen"]
    assert at.metric[3].value == "–" and at.metric[4].value == "–" and all(m.value for m in at.metric)
    at.sidebar.select_slider(key="window_select").set_value(30).run()
    at.sidebar.selectbox(key="select_select").set_value("x").run()
    assert not at.exception and [m.label for m in at.metric][3:5] == ["Mittel über 30 Tage", "Gewählte Standorte"] and at.metric[3].value != "–" and at.metric[4].value != "–"


def test_the_headline_verdict_depends_on_the_world():
    at = _app()
    assert "Dieselben 10 800 Werte" in _texts(at) and "Die Vorzeichen sind verschieden" not in _texts(at)
    at.sidebar.button(key="preset_Gleiche Welt, andere Ebene").click().run()
    assert not at.exception and "Die Vorzeichen sind verschieden" in _texts(at)


def test_the_scatter_view_can_be_changed():
    at = _app()
    at.sidebar.select_slider(key="window_select").set_value(30)
    at.sidebar.selectbox(key="select_select").set_value("x").run()
    for key in ("total", "true", "within", "window", "between", "selected"):
        at.selectbox(key="view_select").set_value(key).run()
        assert not at.exception, (key, [str(e) for e in at.exception])


def test_pdf_button_then_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception and len(at.get("download_button")) == 1


def test_experiment_selectors_run():
    at = _app()
    for key, value in (("e1_sigma", 1.0), ("e1_smooth", 1), ("e1_smooth", 7), ("e2_by", "summe"), ("e2_rho", 0.3), ("e2_rho", 0.8), ("e2_by", "x"), ("e4_case", 1), ("e4_case", 3)):
        at.selectbox(key=key).set_value(value).run()
        assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_the_experiment_sections_show_their_verdicts():
    text = _texts(_app())
    for needle in ("Glättung 30, Rauschen 2", "ρ = 0,5, Auswahl nach X", "Wahres ρ = 0,6", "30 Standorte mit je 100 Tagen"):
        assert needle in text, needle


def test_every_slider_corner_runs():
    for edge in ("min", "max"):
        at = _app()
        for key in ("g_slider", "t_slider", "rho_b_slider", "sigma_slider", "gamma_slider", "q_slider"):
            w = at.sidebar.slider(key=key)
            value = w.min if edge == "min" else w.max
            w.set_value(int(value) if isinstance(w.value, int) else value)
        at.sidebar.select_slider(key="smooth_select").set_value(C.SMOOTHS[0] if edge == "min" else C.SMOOTHS[-1])
        at.sidebar.select_slider(key="window_select").set_value(C.WINDOWS[0] if edge == "min" else C.WINDOWS[-1])
        at.sidebar.selectbox(key="select_select").set_value("summe" if edge == "min" else "x")
        at.run()
        assert not at.exception, (edge, [str(e) for e in at.exception])
        assert all(m.value for m in at.metric)


def test_a_missing_grid_shows_a_warning_instead_of_failing(monkeypatch, tmp_path):
    import streamlit as st

    import agr_evaluation as ev

    st.cache_data.clear()                                  # zwischengespeicherte Raster aus anderen Tests verwerfen
    monkeypatch.setattr(ev, "GRID_PATH", tmp_path / "gibt-es-nicht.json")
    monkeypatch.setattr(ev, "load_grid", lambda path=tmp_path / "gibt-es-nicht.json": ev.json.loads("null"))
    try:
        at = AppTest.from_file(APP, default_timeout=TIMEOUT)
        at.run()
        assert not at.exception, [str(e) for e in at.exception]
        assert any("Das vorgerechnete Raster" in w.value for w in at.warning)
    finally:
        st.cache_data.clear()
