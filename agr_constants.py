"""Konstanten: Regler-Grenzen, Auswahlarten, Raster der Messreihen, Farben und Presets.

Zufallserzeuger und Pearson kommen als wörtliche Kopien aus pearson-demo (prs_rng.py, prs_constants.py, prs_scenario.py, prs_algorithm.py).
Rechenbudget am Regler-Maximum (DEMO-PLAYBOOK §1c): eine Welt mit 60 Standorten und 720 Tagen sind 43 200 Zeilen; alle Zuschnitte (gesamt, Fenster, innerhalb, zwischen, Auswahl) und die exakten
Formeln brauchen zusammen Millisekunden. Die Wiederholungsexperimente stehen vorgerechnet im Raster (data/aggregation_grid.json)."""

from prs_constants import SEED_MAX  # noqa: F401  (Wiederverwendung der Kopie)

G_MIN, G_MAX, DEFAULT_G = 5, 60, 30
T_MIN, T_MAX, DEFAULT_T = 90, 720, 360
RHO_B_MIN, RHO_B_MAX, DEFAULT_RHO_B = -0.8, 0.9, 0.5
SIGMA_MIN, SIGMA_MAX, DEFAULT_SIGMA = 0.0, 3.0, 2.0
SMOOTHS = (1, 7, 30)
DEFAULT_SMOOTH = 30
GAMMA_MIN, GAMMA_MAX, DEFAULT_GAMMA = -1.0, 1.0, 1.0
WINDOWS = (1, 7, 30, 90)
DEFAULT_WINDOW = 1
Q_MIN, Q_MAX, DEFAULT_Q = 5, 100, 20          # Anteil der gewählten Standorte in Prozent
MIN_SELECTED = 3                               # mindestens so viele Standorte in der Auswahl
DEFAULT_SEED = 35

# Auswahl: Schlüssel -> Anzeigename
SELECTIONS = {"keine": "keine Auswahl", "x": "nach X", "summe": "nach X + Y (Summe der standardisierten Werte)"}

# Raster der Messreihen (tools/make_grid.py)
GRID_SEEDS = (101, 102, 103, 104)
WINDOW_T, WINDOW_RUNS = 3600, 50                                       # Tage und Läufe je Seed (Mittelung)
WINDOW_SMOOTHS, WINDOW_SIGMAS = (1, 7, 30), (1.0, 2.0)
SELECT_N, SELECT_RUNS = 400, 500                                       # Standorte und Läufe je Seed (Auswahl nach X)
SELECT_RHOS, SELECT_QS = (0.3, 0.5, 0.8), (1.0, 0.5, 0.2, 0.05)
SUM_N, SUM_RUNS, SUM_QS = 20_000, 10, (0.5, 0.2, 0.1, 0.01)             # Paare und Läufe je Seed (Auswahl nach der Summe)
ATT_N, ATT_RUNS, ATT_RHO = 300, 500, 0.6                               # Standorte, Läufe je Seed, wahres ρ (Messfehler)
ATT_RELS = ((1.0, 1.0), (0.8, 0.8), (0.6, 0.6), (0.5, 0.8), (0.4, 0.4))
LEVEL_RUNS = 200                                                       # Läufe je Seed (Ebenen)
LEVEL_CASES = (                                                        # (G, T, ρ_b, σ, Glättung, γ)
    (30, 100, 0.8, 1.0, 1, -0.5),
    (10, 100, 0.8, 1.0, 1, -0.5),
    (30, 100, 0.0, 1.0, 1, 1.0),
    (30, 100, 0.8, 1.0, 1, 0.5),
)

COLOR_ALL = "#4c78a8"
COLOR_WITHIN = "#54a24b"
COLOR_BETWEEN = "#e45756"
COLOR_SELECT = "#f58518"
COLOR_EXACT = "#222222"
COLOR_GREY = "#9d9d9d"

PRESETS = {
    "Standardfall (Voreinstellung)": {"g": 30, "t": 360, "rho_b": 0.5, "sigma": 2.0, "smooth": 30, "gamma": 1.0, "window": 1, "select": "keine", "q": 20, "seed": DEFAULT_SEED},
    "Tage, Wochen, Monate": {"g": 30, "t": 360, "rho_b": 0.5, "sigma": 2.0, "smooth": 30, "gamma": 1.0, "window": 30, "select": "keine", "q": 20, "seed": DEFAULT_SEED},
    "Welle ohne Gedächtnis": {"g": 30, "t": 360, "rho_b": 0.5, "sigma": 2.0, "smooth": 1, "gamma": 1.0, "window": 30, "select": "keine", "q": 20, "seed": DEFAULT_SEED},
    "Nur die besten Standorte": {"g": 60, "t": 360, "rho_b": 0.5, "sigma": 2.0, "smooth": 30, "gamma": 1.0, "window": 1, "select": "x", "q": 20, "seed": DEFAULT_SEED},
    "Auswahl nach der Summe": {"g": 60, "t": 360, "rho_b": 0.5, "sigma": 2.0, "smooth": 30, "gamma": 1.0, "window": 1, "select": "summe", "q": 20, "seed": DEFAULT_SEED},
    "Gleiche Welt, andere Ebene": {"g": 30, "t": 360, "rho_b": 0.8, "sigma": 1.0, "smooth": 1, "gamma": -0.5, "window": 1, "select": "keine", "q": 20, "seed": DEFAULT_SEED},
}
# Die Hilfetexte nennen exakte Werte und Zahlen des vorgerechneten Rasters; tests/test_claims.py prüft jede davon.
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Eine Welt mit 30 Standorten und 360 Tagen: erwartet sind r = 0,25 über alle Standort-Tage, 0,53 für die Standortmittel und 0,19 innerhalb der Standorte; ohne Messrauschen wären es 0,75.",
    "Tage, Wochen, Monate": "Mittel über 30 Tage heben r innerhalb der Standorte von erwartet 0,19 (Tage) auf 0,83 (Fenster), weil die gemeinsame Welle 30 Tage Gedächtnis hat; aus 10 800 Tageswerten werden 360 Fenstermittel.",
    "Welle ohne Gedächtnis": "Hat die gemeinsame Welle kein Gedächtnis, ändert die Mittelung über 30 Tage nichts: innerhalb der Standorte bleibt r bei erwartet 0,20, Tage wie Fenster.",
    "Nur die besten Standorte": "Die obersten 20 % der 60 Standorte nach X (12 Standorte) zeigen erwartet 0,28 statt 0,53 für alle; mit so wenigen gewählten Standorten streut r stark (Raster: Streuung 0,224 bei 20 gewählten Standorten und ρ = 0,5).",
    "Auswahl nach der Summe": "Die obersten 20 % der 60 Standorte nach X + Y zeigen erwartet -0,17 statt +0,53 für alle: die Auswahl nach der Summe erzeugt eine negative Korrelation aus einer positiven.",
    "Gleiche Welt, andere Ebene": "Erwartet: alle Standort-Tage 0,12, Standortmittel 0,80, innerhalb der Standorte -0,32. Im Raster (30 Standorte, 100 Tage) haben Standortmittel und innerhalb in 100,0 % der 800 Läufe verschiedene Vorzeichen.",
}
