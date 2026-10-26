"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Hilfetexte, Adresszeile."""

import pytest

import agr_constants as C
import agr_evaluation as ev
import agr_presets as P


def test_presets_are_complete():
    assert "Standardfall (Voreinstellung)" in C.PRESETS
    assert set(C.PRESET_HELP) == set(C.PRESETS)
    for name, preset in C.PRESETS.items():
        assert C.PRESET_HELP[name] and C.PRESET_HELP[name] != name         # ein echter Text, nicht der Name
        assert set(preset) == set(P.PRESET_KEYS), name
        assert (C.G_MIN <= preset["g"] <= C.G_MAX and C.T_MIN <= preset["t"] <= C.T_MAX and C.RHO_B_MIN <= preset["rho_b"] <= C.RHO_B_MAX and C.SIGMA_MIN <= preset["sigma"] <= C.SIGMA_MAX
                and preset["smooth"] in C.SMOOTHS and C.GAMMA_MIN <= preset["gamma"] <= C.GAMMA_MAX and preset["window"] in C.WINDOWS and preset["select"] in C.SELECTIONS
                and C.Q_MIN <= preset["q"] <= C.Q_MAX and 0 <= preset["seed"] <= C.SEED_MAX), name


def test_the_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_the_defaults_match():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    s = ev.Settings()
    keys = ("g_slider", "t_slider", "rho_b_slider", "sigma_slider", "smooth_select", "gamma_slider", "window_select", "select_select", "q_slider", "seed_input")
    assert [P.SETTING_SPECS[k].default for k in keys] == [s.g, s.t, s.rho_b, s.sigma, s.smooth, s.gamma, s.window, s.select, s.q, s.seed]


def test_every_preset_differs_from_every_other():
    assert len({tuple(sorted(p.items())) for p in C.PRESETS.values()}) == len(C.PRESETS)


def test_the_presets_use_the_features_they_name():
    assert C.PRESETS["Tage, Wochen, Monate"]["window"] == 30 and C.PRESETS["Welle ohne Gedächtnis"]["smooth"] == 1 and C.PRESETS["Welle ohne Gedächtnis"]["window"] == 30
    assert C.PRESETS["Nur die besten Standorte"]["select"] == "x" and C.PRESETS["Auswahl nach der Summe"]["select"] == "summe"
    assert C.PRESETS["Gleiche Welt, andere Ebene"]["gamma"] < 0 < C.PRESETS["Gleiche Welt, andere Ebene"]["rho_b"]


@pytest.mark.parametrize("raw,expected", [("99999", C.T_MAX), ("1", C.T_MIN), ("200", 200), ("abc", None), ("", None)])
def test_clean_value_clamps_integers(raw, expected):
    assert P.clean_value(P.SETTING_SPECS["t_slider"], raw) == expected


def test_clean_value_accepts_only_known_choices_and_clamps_floats():
    assert P.clean_value(P.SETTING_SPECS["smooth_select"], "7") == 7 and P.clean_value(P.SETTING_SPECS["smooth_select"], "8") is None
    assert P.clean_value(P.SETTING_SPECS["window_select"], "90") == 90 and P.clean_value(P.SETTING_SPECS["window_select"], "x") is None
    assert P.clean_value(P.SETTING_SPECS["select_select"], "summe") == "summe" and P.clean_value(P.SETTING_SPECS["select_select"], "zufall") is None
    assert P.clean_value(P.SETTING_SPECS["rho_b_slider"], "5") == C.RHO_B_MAX and P.clean_value(P.SETTING_SPECS["rho_b_slider"], "-5") == C.RHO_B_MIN
    assert P.clean_value(P.SETTING_SPECS["gamma_slider"], "0.123456") == 0.12
    assert P.clean_value(P.SETTING_SPECS["seed_input"], "-5") == 0 and P.clean_value(P.SETTING_SPECS["seed_input"], str(C.SEED_MAX + 1)) == C.SEED_MAX
