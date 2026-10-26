"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio; wie in den übrigen Stücken der Korrelations-Reihe)."""

import random
from dataclasses import dataclass

import streamlit as st

import agr_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "g_slider": SettingSpec("g", int, C.DEFAULT_G, C.G_MIN, C.G_MAX),
    "t_slider": SettingSpec("t", int, C.DEFAULT_T, C.T_MIN, C.T_MAX),
    "rho_b_slider": SettingSpec("rhob", float, C.DEFAULT_RHO_B, C.RHO_B_MIN, C.RHO_B_MAX),
    "sigma_slider": SettingSpec("sigma", float, C.DEFAULT_SIGMA, C.SIGMA_MIN, C.SIGMA_MAX),
    "smooth_select": SettingSpec("smooth", int, C.DEFAULT_SMOOTH, choices=C.SMOOTHS),
    "gamma_slider": SettingSpec("gamma", float, C.DEFAULT_GAMMA, C.GAMMA_MIN, C.GAMMA_MAX),
    "window_select": SettingSpec("window", int, C.DEFAULT_WINDOW, choices=C.WINDOWS),
    "select_select": SettingSpec("select", str, "keine", choices=tuple(C.SELECTIONS)),
    "q_slider": SettingSpec("q", int, C.DEFAULT_Q, C.Q_MIN, C.Q_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"g": "g_slider", "t": "t_slider", "rho_b": "rho_b_slider", "sigma": "sigma_slider", "smooth": "smooth_select", "gamma": "gamma_slider", "window": "window_select",
               "select": "select_select", "q": "q_slider", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def clean_value(spec: SettingSpec, raw):
    """Wert aus der Adresszeile: umwandeln, auf Reglergrenzen begrenzen, Gleitkommazahlen auf zwei Stellen runden; ungültig gibt None."""
    try:
        value = spec.caster(raw)
    except (ValueError, TypeError):
        return None
    if spec.choices is not None:
        return value if value in spec.choices else None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return round(value, 2) if isinstance(value, float) else value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            value = clean_value(spec, qp[spec.url_param])
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Setzt ALLE Regler: die des Presets auf seine Werte, die übrigen auf ihren Standard (so bleibt die Hervorhebung des Presets nach dem Anklicken gültig)."""
    preset = C.PRESETS[name]
    key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = preset.get(key_of.get(state_key), spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
