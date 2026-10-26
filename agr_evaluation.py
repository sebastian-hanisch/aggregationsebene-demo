"""Auswertung: Einzelwelt, Zellen des vorgerechneten Rasters, Zugriff auf das Raster und alle Verdikte als reine Funktionen."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import agr_algorithm as K
import agr_constants as C
import prs_scenario as S
from prs_algorithm import t_critical

GRID_PATH = Path(__file__).resolve().parent / "data" / "aggregation_grid.json"


# --- Einzelwelt ----------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Settings:
    g: int = C.DEFAULT_G
    t: int = C.DEFAULT_T
    rho_b: float = C.DEFAULT_RHO_B
    sigma: float = C.DEFAULT_SIGMA
    smooth: int = C.DEFAULT_SMOOTH
    gamma: float = C.DEFAULT_GAMMA
    window: int = C.DEFAULT_WINDOW
    select: str = "keine"
    q: int = C.DEFAULT_Q
    seed: int = C.DEFAULT_SEED

    def key(self) -> tuple:
        return (self.g, self.t, self.rho_b, self.sigma, self.smooth, self.gamma, self.window, self.select, self.q, self.seed)

    def __eq__(self, other) -> bool:
        return isinstance(other, Settings) and self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())


@dataclass(frozen=True)
class Row:
    key: str
    name: str
    r: float
    exact: float | None
    points: int


@dataclass(frozen=True, eq=False)
class Analysis:
    settings: Settings
    rows: list                    # Zeilen der Tabelle „Zuschnitt, r, exakt, Punkte“
    data: dict                    # Schlüssel -> (x, y) der Werte, aus denen die Zeile gerechnet ist
    selected: np.ndarray | None   # Zeilennummern der gewählten Standorte (bei Auswahl)
    split: dict                   # Zerlegung gesamt, zwischen, innerhalb mit den Kreuzproduktsummen
    rx: float                     # Zuverlässigkeit der Tageswerte
    ry: float


def analyse(s: Settings) -> Analysis:
    data_seed = S.substream_seeds(s.seed, 2)[0]
    X, Y, X0, Y0 = K.make_world(s.g, s.t, s.rho_b, s.sigma, s.smooth, s.gamma, S.VRng(data_seed))
    split = K.split_levels(X, Y)
    dx, dy = X - X.mean(axis=1, keepdims=True), Y - Y.mean(axis=1, keepdims=True)
    mx, my = X.mean(axis=1), Y.mean(axis=1)
    rx, ry = K.reliabilities(s.sigma, s.gamma)
    rows, data = [], {}

    def add(key: str, name: str, x, y, r: float, exact: float | None) -> None:
        rows.append(Row(key, name, r, exact, int(np.size(x))))
        data[key] = (np.asarray(x).ravel(), np.asarray(y).ravel())

    add("total", "Alle Standort-Tage", X, Y, split["total"], K.exact_pooled_r(1, s.smooth, s.sigma, s.gamma, s.rho_b))
    add("true", "Alle Standort-Tage ohne Messrauschen (wahre Werte)", X0, Y0, K.r_all(X0, Y0), K.exact_true_r(s.smooth, s.gamma, s.rho_b))
    if s.sigma > 0:
        rows.append(Row("corrected", "Alle Standort-Tage, korrigiert für das Messrauschen", K.disattenuate(split["total"], rx, ry), K.exact_true_r(s.smooth, s.gamma, s.rho_b), s.g * s.t))
    add("within", "Innerhalb der Standorte (Abweichung vom Standortmittel)", dx, dy, split["within"], K.exact_within_r(s.t, s.smooth, s.sigma, s.gamma))
    if s.window > 1 and s.t // s.window >= 2:                      # bei nur einem Fenster je Standort ist die Abweichung vom Standortmittel null
        wx, wy = K.window_means(dx, s.window), K.window_means(dy, s.window)
        add("window", f"Innerhalb, Mittel über {s.window} Tage", wx, wy, K.r_all(wx, wy), K.exact_within_window_r(s.t, s.window, s.smooth, s.sigma, s.gamma))
    exact_between = K.exact_pooled_r(s.t, s.smooth, s.sigma, s.gamma, s.rho_b)
    add("between", "Standortmittel (zwischen den Standorten)", mx, my, split["between"], exact_between)
    selected = None
    if s.select != "keine":
        selected = K.select_top(mx, my, s.q / 100.0, s.select)
        k = len(selected)
        name = f"Standortmittel der obersten {s.q} % ({k} Standorte) nach {'X' if s.select == 'x' else 'X + Y'}"
        add("selected", name, mx[selected], my[selected], K.pearson_r(mx[selected], my[selected]), K.exact_select(exact_between, k / s.g, s.select))
    return Analysis(s, rows, data, selected, split, rx, ry)


def row(a: Analysis, key: str) -> Row | None:
    return next((r for r in a.rows if r.key == key), None)


# --- Zellen des vorgerechneten Rasters -------------------------------------------------------------------------------------------------------

def cell_seed(kind: str, a: float, b: float, c: float, s: int) -> int:
    """Eigener Seed je Zelle und Plan-Seed s (verschieden für Art und die drei Parameter der Zelle)."""
    kinds = ("window", "select", "attenuation", "level", "probe")
    return s * 1_000_003 + int(round(a * 1000)) * 7919 + int(round(b * 1000)) * 1013 + int(round(c * 1000)) * 101 + kinds.index(kind) * 100_000_007


def _sd(values) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _r_crit(n: int) -> float:
    t = t_critical(n - 2)
    return t / math.sqrt(t * t + n - 2)


def window_cell(smooth: int, sigma: float, m: int, runs_per_seed: int = C.WINDOW_RUNS, seeds=C.GRID_SEEDS, T: int = C.WINDOW_T) -> dict:
    """E1: eine Welle (kein Standortniveau) mit Gedächtnis `smooth`, x = Welle + σ ε, y = Welle + σ η; r der Fenstermittel (Mittel, Streuung, Anteil „signifikant“) gegen den exakten Wert."""
    rs = []
    for s in seeds:
        rng = S.VRng(cell_seed("window", smooth, sigma, m, s))
        for _ in range(runs_per_seed):
            w = K.wave(1, T, smooth, rng)[0]
            a, b = w + sigma * rng.normal(T), w + sigma * rng.normal(T)
            rs.append(K.pearson_r(K.window_means(a, m), K.window_means(b, m)))
    rs = np.array(rs)
    n = T // m
    return {"smooth": smooth, "sigma": sigma, "m": m, "points": n, "runs": len(rs), "mean": float(rs.mean()), "sd": _sd(rs), "exact": K.exact_wave_window_r(m, smooth, sigma), "sig": float((np.abs(rs) > _r_crit(n)).mean())}


def select_cell(by: str, rho: float, q: float, n: int, runs_per_seed: int, seeds=C.GRID_SEEDS) -> dict:
    """E2: Auswahl der oberen Quote q von n Einheiten nach X ("x") oder nach der Summe ("summe"); r in der Auswahl (Mittel, Streuung, Anteil negativ, Anteil signifikant positiv) gegen die exakte Formel."""
    k = K.select_count(n, q)
    rs = []
    for s in seeds:
        rng = S.VRng(cell_seed("select", rho, q, 0 if by == "x" else 1, s))
        for _ in range(runs_per_seed):
            x, y = K.draw_pairs(n, rho, rng)
            idx = K.select_top(x, y, q, by)
            rs.append(K.pearson_r(x[idx], y[idx]))
    rs = np.array(rs)
    return {"by": by, "rho": rho, "q": q, "n": n, "k": k, "runs": len(rs), "mean": float(rs.mean()), "sd": _sd(rs), "neg": float((rs < 0).mean()), "sig_pos": float((rs > _r_crit(k)).mean()), "exact": K.exact_select(rho, k / n, by)}


def attenuation_cell(rx: float, ry: float, runs_per_seed: int = C.ATT_RUNS, seeds=C.GRID_SEEDS, n: int = C.ATT_N, rho: float = C.ATT_RHO) -> dict:
    """E3: wahres ρ, beobachtet mit Zuverlässigkeit rx, ry: r (Mittel, Streuung), korrigiertes r (Mittel, Streuung) gegen die exakte Verdünnung."""
    rs = []
    for s in seeds:
        rng = S.VRng(cell_seed("attenuation", rx, ry, rho, s))
        for _ in range(runs_per_seed):
            a, b = K.noisy_pairs(n, rho, rx, ry, rng)
            rs.append(K.pearson_r(a, b))
    rs = np.array(rs)
    corrected = rs / math.sqrt(rx * ry)
    return {"rx": rx, "ry": ry, "rho": rho, "n": n, "runs": len(rs), "exact": K.attenuated_r(rho, rx, ry), "mean": float(rs.mean()), "sd": _sd(rs), "corrected_mean": float(corrected.mean()), "corrected_sd": _sd(corrected)}


def level_cell(g: int, t: int, rho_b: float, sigma: float, smooth: int, gamma: float, runs_per_seed: int = C.LEVEL_RUNS, seeds=C.GRID_SEEDS) -> dict:
    """E4: eine Welt, drei Antworten: r über alle Standort-Tage, der Standortmittel und innerhalb (Mittel und Streuung über die Läufe), mit den erwarteten Werten und dem Anteil der Läufe,
    in denen Vorzeichen von zwischen und innerhalb verschieden sind."""
    tot, bet, wit = [], [], []
    for s in seeds:
        rng = S.VRng(cell_seed("level", rho_b, sigma, gamma + g / 1000 + t / 1_000_000, s))
        for _ in range(runs_per_seed):
            X, Y, _x0, _y0 = K.make_world(g, t, rho_b, sigma, smooth, gamma, rng)
            sp = K.split_levels(X, Y)
            tot.append(sp["total"])
            bet.append(sp["between"])
            wit.append(sp["within"])
    tot, bet, wit = np.array(tot), np.array(bet), np.array(wit)
    return {"g": g, "t": t, "rho_b": rho_b, "sigma": sigma, "smooth": smooth, "gamma": gamma, "runs": len(tot),
            "total": float(tot.mean()), "total_sd": _sd(tot), "between": float(bet.mean()), "between_sd": _sd(bet), "within": float(wit.mean()), "within_sd": _sd(wit),
            "exact_total": K.exact_pooled_r(1, smooth, sigma, gamma, rho_b), "exact_between": K.exact_pooled_r(t, smooth, sigma, gamma, rho_b), "exact_within": K.exact_within_r(t, smooth, sigma, gamma),
            "sign_differs": float((np.sign(bet) != np.sign(wit)).mean())}


def grid_probe() -> dict:
    """Kleine feste Zellen und eine feste Einzelwelt, die der Test nachrechnet: ändert sich Erzeuger oder Kern, passen sie nicht mehr zum gespeicherten Raster."""
    a = analyse(Settings(12, 120, 0.5, 1.0, 7, 0.5, 7, "x", 50, 11))
    return {"window": window_cell(7, 1.0, 7, 3, (7,), 600), "select_x": select_cell("x", 0.5, 0.2, 100, 5, (7,)), "select_sum": select_cell("summe", 0.5, 0.1, 500, 3, (7,)),
            "attenuation": attenuation_cell(0.6, 0.8, 5, (7,), 100), "level": level_cell(8, 40, 0.8, 1.0, 1, -0.5, 4, (7,)),
            "rows": {r.key: [r.r, r.exact, r.points] for r in a.rows}}


# --- Zugriff auf das Raster ------------------------------------------------------------------------------------------------------------------

def load_grid(path: Path = GRID_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find(rows: list[dict], **keys) -> dict | None:
    for r in rows:
        if all(r.get(k) == v or (isinstance(v, float) and isinstance(r.get(k), (int, float)) and abs(r[k] - v) < 1e-9) for k, v in keys.items()):
            return r
    return None


# --- Verdikte (reine Funktionen: Art, Text) -----------------------------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{100 * v:.1f} %".replace(".", ",")


def _num(v: float, digits: int = 2) -> str:
    return f"{v:.{digits}f}".replace(".", ",")


def _int(v: int) -> str:
    """Ganze Zahl mit Leerzeichen als Tausendertrenner."""
    return f"{v:,}".replace(",", " ")


def _n(v: float) -> str:
    """Zahl ohne unnötige Stellen (0,5; 0,99; 2)."""
    return f"{v:g}".replace(".", ",")


def verdict_main(a: Analysis) -> tuple[str, str]:
    """Kernfrage der Hauptansicht: eine Welt, drei Antworten (gesamt, zwischen den Standorten, innerhalb der Standorte)."""
    total, between, within = row(a, "total"), row(a, "between"), row(a, "within")
    text = (f"Dieselben {_int(a.settings.g * a.settings.t)} Werte, drei Antworten: alle Standort-Tage **{_num(total.r)}**, die Standortmittel **{_num(between.r)}**, innerhalb der Standorte **{_num(within.r)}**. "
            f"Erwartet wären {_num(total.exact)}, {_num(between.exact)} und {_num(within.exact)}.")
    differs = (between.r > 0) != (within.r > 0) and min(abs(between.r), abs(within.r)) > 0.05
    if differs:
        return "warning", text + " **Die Vorzeichen sind verschieden**: wer die Standorte vergleicht, sieht das Gegenteil dessen, was innerhalb der Standorte gilt."
    return ("warning" if abs(between.r - within.r) > 0.3 else "info"), text


def verdict_window(a: Analysis) -> tuple[str, str] | None:
    """Mittelung: r der Fenstermittel gegen r der Tage innerhalb der Standorte (nur bei Fenster > 1)."""
    w, within = row(a, "window"), row(a, "within")
    s = a.settings
    if w is None:
        if s.window > 1:
            return "warning", f"Bei {s.t} Tagen gibt es für das Fenster von {s.window} Tagen weniger als zwei volle Fenster je Standort; die Mittelung innerhalb der Standorte ist dann nicht definiert (Fenster höchstens T/2)."
        return None
    text = (f"Mittel über {s.window} Tage statt einzelner Tage: r innerhalb der Standorte steigt von **{_num(within.r)}** auf **{_num(w.r)}** (erwartet {_num(w.exact)}), "
            f"aber aus {_int(within.points)} Tagen werden {_int(w.points)} Fenster.")
    if s.smooth == 1:
        text += " Die Welle hat kein Gedächtnis: dann ändert die Mittelung r kaum."
    return "info", text


def verdict_selection(a: Analysis) -> tuple[str, str] | None:
    """Auswahl: r der gewählten Standorte gegen r aller Standortmittel (nur bei Auswahl)."""
    sel, between = row(a, "selected"), row(a, "between")
    if sel is None:
        return None
    kind = "nach X" if a.settings.select == "x" else "nach der Summe X + Y"
    text = f"Nur die obersten {a.settings.q} % {kind} ({sel.points} Standorte): r **{_num(sel.r)}** statt {_num(between.r)} bei allen {between.points} (erwartet {_num(sel.exact)})."
    if sel.r < 0 < between.r:
        text += " **Die Auswahl hat das Vorzeichen gekehrt.**"
    return ("warning" if abs(sel.r - between.r) > 0.2 else "info"), text


def verdict_noise(a: Analysis) -> tuple[str, str] | None:
    """Messrauschen: r ohne Rauschen gegen r mit Rauschen und die korrigierte Zahl (nur bei σ > 0)."""
    total, true, corr = row(a, "total"), row(a, "true"), row(a, "corrected")
    if corr is None:
        return None
    return "info", (f"Ohne Messrauschen wäre r **{_num(true.r)}** statt {_num(total.r)}; die Zuverlässigkeit der Tageswerte ist {_num(a.rx)} und {_num(a.ry)}, die Korrektur gibt **{_num(corr.r)}** "
                    f"(erwartet {_num(corr.exact)}); sie braucht die Zuverlässigkeit und streut stärker.")


def verdict_window_grid(rows: list[dict], smooth: int, sigma: float) -> tuple[str, str]:
    """E1: r gegen das Fenster bei einer Glättung und einem Rauschen; Punkte und Streuung am größten Fenster."""
    cells = sorted((r for r in rows if r["smooth"] == smooth and r["sigma"] == sigma), key=lambda r: r["m"])
    first, last = cells[0], cells[-1]
    mid = find(cells, m=7) or cells[1]
    text = (f"Glättung {smooth}, Rauschen {_n(sigma)}: r ist **{_num(first['mean'], 3)}** bei Tagen, **{_num(mid['mean'], 3)}** bei {mid['m']} Tagen und **{_num(last['mean'], 3)}** bei {last['m']} Tagen "
            f"(exakt {_num(first['exact'], 3)} / {_num(mid['exact'], 3)} / {_num(last['exact'], 3)}); aus {_int(first['points'])} Punkten werden {_int(last['points'])}, die Streuung von r steigt von {_num(first['sd'], 3)} auf {_num(last['sd'], 3)}, "
            f"der Anteil signifikanter Läufe fällt von {_pct(first['sig'])} auf {_pct(last['sig'])}.")
    return ("info" if smooth > 1 else "warning"), text


def verdict_select_grid(rows: list[dict], by: str, rho: float) -> tuple[str, str]:
    """E2: r nach Auswahl bei einem ρ: erste und letzte Quote, exakt daneben."""
    cells = sorted((r for r in rows if r["by"] == by and r["rho"] == rho), key=lambda r: -r["q"])
    first, last = cells[0], cells[-1]
    what = "nach X" if by == "x" else "nach der Summe"
    text = (f"ρ = {_n(rho)}, Auswahl {what}: die obersten {_pct(first['q'])} der Einheiten zeigen **{_num(first['mean'], 3)}** (exakt {_num(first['exact'], 3)}), "
            f"die obersten {_pct(last['q'])} **{_num(last['mean'], 3)}** (exakt {_num(last['exact'], 3)}).")
    if by == "x":
        text += f" Bei nur {last['k']} gewählten Einheiten ist r in {_pct(last['neg'])} der Läufe negativ und nur in {_pct(last['sig_pos'])} signifikant positiv."
    elif last["mean"] < 0 < rho:
        text += " **Die Auswahl erzeugt eine negative Korrelation aus einer positiven.**"
    return "warning", text


def verdict_attenuation_grid(rows: list[dict]) -> tuple[str, str]:
    """E3: Verdünnung und Korrektur an der Zeile mit der kleinsten Zuverlässigkeit."""
    low = min(rows, key=lambda r: r["rx"] * r["ry"])
    top = max(rows, key=lambda r: r["rx"] * r["ry"])
    return "info", (f"Wahres ρ = {_n(top['rho'])}: bei Zuverlässigkeit {_n(low['rx'])} und {_n(low['ry'])} bleibt r **{_num(low['mean'], 3)}** (exakt {_num(low['exact'], 3)}); die Korrektur trifft im Mittel "
                    f"**{_num(low['corrected_mean'], 3)}**, aber ihre Streuung ist {_num(low['corrected_sd'], 3)} statt {_num(top['corrected_sd'], 3)} ohne Messfehler.")


def verdict_levels_grid(rows: list[dict], index: int = 0) -> tuple[str, str]:
    """E4: eine Zeile der Ebenen-Tabelle: gesamt, zwischen und innerhalb mit den Vorzeichen."""
    c = rows[index]
    text = (f"{c['g']} Standorte mit je {c['t']} Tagen, ρ_b = {_n(c['rho_b'])}, Kopplung {_n(c['gamma'])}: alle Standort-Tage **{_num(c['total'], 3)}**, Standortmittel **{_num(c['between'], 3)}** (Streuung {_num(c['between_sd'], 3)}), "
            f"innerhalb **{_num(c['within'], 3)}**; erwartet {_num(c['exact_total'], 3)}, {_num(c['exact_between'], 3)}, {_num(c['exact_within'], 3)}.")
    if c["sign_differs"] > 0.5:
        text += f" In {_pct(c['sign_differs'])} der Läufe haben Standortmittel und innerhalb verschiedene Vorzeichen."
    return "warning", text
