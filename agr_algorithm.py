"""Kern: eine Welt aus Standorten und Tagen (Standortniveau, gemeinsame Welle mit Gedächtnis, Tagesrauschen), Mittelung über Fenster, Auswahl nach X und nach der Summe, Zerlegung der Korrelation
in gesamt, zwischen und innerhalb, Messfehler-Verdünnung und ihre Korrektur, dazu die exakten Formeln (Erwartung) für alle Zuschnitte.

Welt: x_gt = μx_g + w_gt + σ ε_gt, y_gt = μy_g + γ w_gt + σ η_gt mit Standortniveaus (μx, μy) (Standardnormal, Korrelation ρ_b), einer Welle w (Varianz 1, gleitendes Mittel über `smooth` Tage,
Autokovarianz max(0, 1 − |h|/smooth)) und Rauschen ε, η (Standardnormal). Zufall und Pearson kommen aus den wörtlichen Kopien von pearson-demo (prs_*.py). Ohne scipy zur Laufzeit."""

import math

import numpy as np

import agr_constants as C
from prs_algorithm import norm_ppf, pearson_r


# --- Welt ------------------------------------------------------------------------------------------------------------------------------

def wave(G: int, T: int, smooth: int, rng) -> np.ndarray:
    """G Wellen der Länge T mit Varianz 1: gleitendes Mittel über `smooth` Tage von weißem Rauschen, mal √smooth; Autokovarianz max(0, 1 − |h|/smooth)."""
    if smooth < 1:
        raise ValueError("smooth muss mindestens 1 sein")
    z = rng.normal((G, T + smooth - 1))
    c = np.concatenate([np.zeros((G, 1)), np.cumsum(z, axis=1)], axis=1)
    return (c[:, smooth:] - c[:, :-smooth]) / math.sqrt(smooth)


def make_world(G: int, T: int, rho_b: float, sigma: float, smooth: int, gamma: float, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(X, Y, X0, Y0), jedes G × T: beobachtete Werte und die wahren Werte ohne Rauschen."""
    if not -1.0 < rho_b < 1.0:
        raise ValueError("rho_b muss zwischen −1 und 1 liegen")
    if G < 2 or T < 2:
        raise ValueError("mindestens zwei Standorte und zwei Tage nötig")
    mx = rng.normal(G)
    my = rho_b * mx + math.sqrt(1.0 - rho_b * rho_b) * rng.normal(G)
    w = wave(G, T, smooth, rng)
    X0 = mx[:, None] + w
    Y0 = my[:, None] + gamma * w
    eps, eta = rng.normal((G, T)), rng.normal((G, T))
    return X0 + sigma * eps, Y0 + sigma * eta, X0, Y0


def window_means(A, m: int) -> np.ndarray:
    """Mittel über je m aufeinanderfolgende Tage (entlang der letzten Achse); ein Rest von weniger als m Tagen am Ende wird verworfen."""
    A = np.asarray(A, dtype=float)
    if m < 1 or m > A.shape[-1]:
        raise ValueError("das Fenster muss zwischen 1 und der Zahl der Tage liegen")
    n = A.shape[-1] // m * m
    return A[..., :n].reshape(*A.shape[:-1], n // m, m).mean(axis=-1)


# --- Zerlegung --------------------------------------------------------------------------------------------------------------------------

def r_all(A, B) -> float:
    """Pearson-Korrelation über alle Werte (flach)."""
    return pearson_r(np.asarray(A, dtype=float).ravel(), np.asarray(B, dtype=float).ravel())


def split_levels(X, Y) -> dict:
    """Gesamt, zwischen (Standortmittel) und innerhalb (Abweichungen vom Standortmittel): Korrelationen und die Kreuzproduktsummen der Zerlegung
    Σ(x − x̄)(y − ȳ) = Σ innerhalb + T · Σ_g (x̄_g − x̄)(ȳ_g − ȳ)."""
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    G, T = X.shape
    mx, my = X.mean(axis=1), Y.mean(axis=1)
    dx, dy = X - mx[:, None], Y - my[:, None]
    gx, gy = X.mean(), Y.mean()
    total = float(np.sum((X - gx) * (Y - gy)))
    within = float(np.sum(dx * dy))
    between = float(T * np.sum((mx - gx) * (my - gy)))
    return {"total": r_all(X, Y), "between": pearson_r(mx, my), "within": r_all(dx, dy), "ss_total": total, "ss_within": within, "ss_between": between}


# --- Auswahl ------------------------------------------------------------------------------------------------------------------------------

def standardize(v) -> np.ndarray:
    """z-Werte; ein konstanter Vektor ergibt lauter Nullen."""
    v = np.asarray(v, dtype=float)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def select_count(n: int, q: float) -> int:
    """Zahl der gewählten Einheiten bei der oberen Quote q (Anteil 0 bis 1), mindestens C.MIN_SELECTED, höchstens n."""
    return min(n, max(C.MIN_SELECTED, int(round(n * q))))


def select_top(x, y, q: float, by: str) -> np.ndarray:
    """Zeilennummern der oberen Quote q nach x ("x") oder nach der Summe der standardisierten Werte ("summe")."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if by == "x":
        score = x
    elif by == "summe":
        score = standardize(x) + standardize(y)
    else:
        raise ValueError(f"unbekannte Auswahl: {by}")
    return np.argsort(-score, kind="stable")[:select_count(len(x), q)]


def trunc_variance_ratio(q: float) -> float:
    """Varianzverhältnis u der oberen Quote q einer Standardnormalverteilung (Varianz nach der Auswahl geteilt durch Varianz davor): 1 + λc − λ², λ = φ(c)/q, c = Φ⁻¹(1 − q)."""
    if not 0.0 < q <= 1.0:
        raise ValueError("q muss in (0, 1] liegen")
    if q == 1.0:
        return 1.0
    c = norm_ppf(1.0 - q)
    lam = math.exp(-0.5 * c * c) / math.sqrt(2.0 * math.pi) / q
    return 1.0 + lam * c - lam * lam


def exact_select_x(rho: float, q: float) -> float:
    """Korrelation nach Auswahl der oberen Quote q nach X (Thorndike): ρ √u / √(1 − ρ² + ρ² u)."""
    u = trunc_variance_ratio(q)
    return rho * math.sqrt(u) / math.sqrt(1.0 - rho * rho + rho * rho * u)


def exact_select_sum(rho: float, q: float) -> float:
    """Korrelation nach Auswahl der oberen Quote q nach der Summe zweier gleich streuender Größen: D = X − Y bleibt unberührt, S wird gestutzt: (V_S − V_D)/(V_S + V_D)
    mit V_S = (2 + 2ρ) u und V_D = 2 − 2ρ."""
    vs, vd = (2.0 + 2.0 * rho) * trunc_variance_ratio(q), 2.0 - 2.0 * rho
    return (vs - vd) / (vs + vd)


def exact_select(rho: float, q: float, by: str) -> float:
    return exact_select_x(rho, q) if by == "x" else exact_select_sum(rho, q)


# --- Exakte Formeln der Welt ----------------------------------------------------------------------------------------------------------------

def wave_mean_variance(m: int, smooth: int) -> float:
    """Varianz des Mittels über m aufeinanderfolgende Tage der Welle: (m + 2 Σ_{h<m} (m − h) γ(h))/m² mit γ(h) = max(0, 1 − |h|/smooth)."""
    return (m + 2.0 * sum((m - h) * max(0.0, 1.0 - h / smooth) for h in range(1, m))) / (m * m)


def exact_pooled_r(m: int, smooth: int, sigma: float, gamma: float, rho_b: float) -> float:
    """Erwartete Korrelation über alle Standort-Fenster (Mittel über m Tage, Standortniveaus dabei): (ρ_b + γ V_m)/√((1 + V_m + σ²/m)(1 + γ² V_m + σ²/m)).
    Für m = 1 die Korrelation über alle Standort-Tage, für m = T die der Standortmittel."""
    v = wave_mean_variance(m, smooth)
    return (rho_b + gamma * v) / math.sqrt((1.0 + v + sigma * sigma / m) * (1.0 + gamma * gamma * v + sigma * sigma / m))


def exact_wave_window_r(m: int, smooth: int, sigma: float, gamma: float = 1.0) -> float:
    """Erwartete Korrelation der Fenstermittel ohne Standortniveaus (nur Welle plus Rauschen): γ V_m/√((V_m + σ²/m)(γ² V_m + σ²/m)); bei γ = 1 gleich V_m/(V_m + σ²/m)."""
    v = wave_mean_variance(m, smooth)
    return gamma * v / math.sqrt((v + sigma * sigma / m) * (gamma * gamma * v + sigma * sigma / m))


def exact_within_r(T: int, smooth: int, sigma: float, gamma: float) -> float:
    """Erwartete Korrelation der Abweichungen der Tage vom Standortmittel: Kovarianz γ(1 − V_T), Varianzen (1 + σ²) − (V_T + σ²/T) bzw. (γ² + σ²) − (γ² V_T + σ²/T)."""
    v = wave_mean_variance(T, smooth)
    vx = 1.0 + sigma * sigma - v - sigma * sigma / T
    vy = gamma * gamma + sigma * sigma - gamma * gamma * v - sigma * sigma / T
    return gamma * (1.0 - v) / math.sqrt(vx * vy)


def exact_within_window_r(T: int, m: int, smooth: int, sigma: float, gamma: float) -> float:
    """Erwartete Korrelation der Fenstermittel der Abweichungen vom Standortmittel (Mittel über alle T Tage; ein Rest hinter dem letzten vollen Fenster zählt nur im Standortmittel mit).
    Je Fenster k: Kovarianz γ (V_m − 2 c_k + V_T), Varianzen (V_m − 2 c_k + V_T) + σ² (1/m − 1/T) bzw. γ² (…) + σ² (1/m − 1/T), dabei c_k die Kovarianz des Fenstermittels der Welle mit
    dem Mittel über alle Tage; gemittelt über die Fenster. Für m = 1 gleich `exact_within_r`."""
    if m < 1 or T // m < 2:
        raise ValueError("das Fenster braucht mindestens zwei volle Fenster je Standort (m ≤ T/2)")
    h = np.arange(T)
    acov = np.maximum(0.0, 1.0 - h / smooth)
    # S_i = Σ_j γ(|i − j|) für i = 0 … T − 1
    full = np.concatenate([acov[:0:-1], acov])                      # γ(−(T−1)) … γ(T−1)
    row_sums = np.array([full[T - 1 - i:2 * T - 1 - i].sum() for i in range(T)])
    v_t = float(row_sums.sum()) / (T * T)
    v_m = wave_mean_variance(m, smooth)
    k = T // m
    c = row_sums[:k * m].reshape(k, m).sum(axis=1) / (m * T)
    core = v_m - 2.0 * c + v_t
    noise = sigma * sigma * (1.0 / m - 1.0 / T)
    cov = gamma * float(core.mean())
    vx = float(core.mean()) + noise
    vy = gamma * gamma * float(core.mean()) + noise
    return cov / math.sqrt(vx * vy)


def exact_true_r(smooth: int, gamma: float, rho_b: float) -> float:
    """Erwartete Korrelation über alle Standort-Tage der wahren Werte (ohne Rauschen)."""
    return exact_pooled_r(1, smooth, 0.0, gamma, rho_b)


def reliabilities(sigma: float, gamma: float) -> tuple[float, float]:
    """Zuverlässigkeit der Tageswerte (Anteil der wahren Varianz an der beobachteten, Standortniveau und Welle zusammen): rx = 2/(2 + σ²), ry = (1 + γ²)/(1 + γ² + σ²)."""
    return 2.0 / (2.0 + sigma * sigma), (1.0 + gamma * gamma) / (1.0 + gamma * gamma + sigma * sigma)


def attenuated_r(rho: float, rx: float, ry: float) -> float:
    """Verdünnung durch Messfehler (Spearman): beobachtetes r = ρ √(rx ry)."""
    return rho * math.sqrt(rx * ry)


def disattenuate(r: float, rx: float, ry: float) -> float:
    """Korrektur der Verdünnung: r / √(rx ry); die Zuverlässigkeiten müssen bekannt sein."""
    if rx <= 0.0 or ry <= 0.0:
        raise ValueError("die Zuverlässigkeiten müssen positiv sein")
    return r / math.sqrt(rx * ry)


def expected_total_r(rho_b: float, rho_w: float, sigma_b: float = 1.0, sigma_w: float = 1.0) -> float:
    """Erwartete Gesamtkorrelation bei Standortniveaus (σ_b, ρ_b) und Tagesabweichungen (σ_w, ρ_w): (ρ_b σ_b² + ρ_w σ_w²)/(σ_b² + σ_w²)."""
    return (rho_b * sigma_b ** 2 + rho_w * sigma_w ** 2) / (sigma_b ** 2 + sigma_w ** 2)


# --- Einfache Paare für die Experimente ---------------------------------------------------------------------------------------------------

def draw_pairs(n: int, rho: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """n Paare aus einer zweidimensionalen Normalverteilung (Standardnormal, Korrelation rho)."""
    x = rng.normal(n)
    return x, rho * x + math.sqrt(1.0 - rho * rho) * rng.normal(n)


def noisy_pairs(n: int, rho: float, rx: float, ry: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Beobachtete Werte mit Messfehler: wahre Paare (Korrelation rho) plus Fehler, so dass der Anteil der wahren Varianz rx bzw. ry ist."""
    tx, ty = draw_pairs(n, rho, rng)
    return tx + math.sqrt((1.0 - rx) / rx) * rng.normal(n), ty + math.sqrt((1.0 - ry) / ry) * rng.normal(n)
