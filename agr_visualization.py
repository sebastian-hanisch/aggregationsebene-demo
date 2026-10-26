"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import agr_constants as C

ROW_SHORT = {"total": "Alle Standort-Tage", "true": "… ohne Messrauschen", "corrected": "… korrigiert", "within": "Innerhalb der Standorte", "window": "Innerhalb, Fenster", "between": "Standortmittel", "selected": "Gewählte Standorte"}
ROW_COLORS = {"total": C.COLOR_ALL, "true": C.COLOR_GREY, "corrected": C.COLOR_GREY, "within": C.COLOR_WITHIN, "window": C.COLOR_WITHIN, "between": C.COLOR_BETWEEN, "selected": C.COLOR_SELECT}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", yanchor="top", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _fmt(v: float) -> str:
    return f"{v:.3f}".replace(".", ",")


def build_scatter(x, y, r: float, name: str, color: str, selected=None, max_points: int = 3000, height: int = 360):
    """Streudiagramm der Werte eines Zuschnitts; bei einer Auswahl sind die gewählten Punkte hervorgehoben. Mehr als `max_points` Punkte werden gleichmäßig ausgedünnt (nur die Anzeige)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.zeros(len(x), dtype=bool)
    if selected is not None:
        mask[np.asarray(selected, dtype=int)] = True
    step = max(1, len(x) // max_points)
    keep = np.arange(len(x)) % step == 0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[keep & ~mask].tolist(), y=y[keep & ~mask].tolist(), mode="markers", name="alle" if selected is not None else "Werte", marker=dict(size=5, color=color if selected is None else C.COLOR_GREY, opacity=0.45),
                             hovertemplate="X %{x:.2f}, Y %{y:.2f}<extra></extra>"))
    if mask.any():
        fig.add_trace(go.Scatter(x=x[mask].tolist(), y=y[mask].tolist(), mode="markers", name="gewählte Standorte", marker=dict(size=9, color=C.COLOR_SELECT, line=dict(width=1, color="#ffffff")),
                                 hovertemplate="X %{x:.2f}, Y %{y:.2f}<extra>gewählt</extra>"))
    if len(x) > 2 and float(np.std(x)) > 0:
        xs = np.array([float(x.min()), float(x.max())])
        slope = float(np.polyfit(x, y, 1)[0])
        fig.add_trace(go.Scatter(x=xs.tolist(), y=(float(y.mean()) + slope * (xs - float(x.mean()))).tolist(), mode="lines", name=f"Gerade (r = {_fmt(r)})", line=dict(color=C.COLOR_EXACT, width=2, dash="dash"), hoverinfo="skip"))
    fig.update_xaxes(title_text="Kennzahl X")
    fig.update_yaxes(title_text="Kennzahl Y")
    fig.update_layout(title=dict(text=name, x=0.5, font=dict(size=13)))
    fig = _base(fig, height)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def build_rows_bar(rows: list, height: int = 360):
    """Alle Zuschnitte der Hauptansicht als Balken (gemessen) mit den erwarteten Werten als Rauten; die Achsenbeschriftung ist gekürzt, der volle Name steht im Hover."""
    short = [ROW_SHORT.get(r.key, r.name) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=short, x=[r.r for r in rows], orientation="h", name="gemessen", marker_color=[ROW_COLORS.get(r.key, C.COLOR_ALL) for r in rows], text=[_fmt(r.r) for r in rows], textposition="outside",
                         customdata=[r.name for r in rows], hovertemplate="%{customdata}: %{x:.3f}<extra>gemessen</extra>"))
    with_exact = [(s, r) for s, r in zip(short, rows) if r.exact is not None]
    fig.add_trace(go.Scatter(y=[s for s, _ in with_exact], x=[r.exact for _, r in with_exact], mode="markers", name="erwartet (exakt)", marker=dict(size=12, symbol="diamond-open", color=C.COLOR_EXACT, line=dict(width=2)),
                             hovertemplate="%{y}: %{x:.3f}<extra>erwartet</extra>"))
    fig.update_xaxes(title_text="Korrelation r", range=[-1.15, 1.15])
    fig.update_yaxes(autorange="reversed", automargin=True)
    return _base(fig, height)


def build_window(rows: list, sigma: float, height: int = 360):
    """E1: r gegen das Fenster (Tage) für die drei Glättungen der Welle bei einem Rauschen: gemessen (Linie, Streuung als Balken) und exakt (Rauten)."""
    fig = go.Figure()
    colors = {1: C.COLOR_GREY, 7: C.COLOR_ALL, 30: C.COLOR_WITHIN}
    for smooth in sorted({r["smooth"] for r in rows}):
        cells = sorted((r for r in rows if r["smooth"] == smooth and r["sigma"] == sigma), key=lambda r: r["m"])
        xs = [str(r["m"]) for r in cells]
        label = "kein Gedächtnis" if smooth == 1 else f"Gedächtnis {smooth} Tage"
        fig.add_trace(go.Scatter(x=xs, y=[r["mean"] for r in cells], mode="lines+markers", name=label, line=dict(color=colors.get(smooth, C.COLOR_ALL), width=3),
                                 error_y=dict(type="data", array=[r["sd"] for r in cells], visible=True, thickness=1.5, width=4), hovertemplate="Fenster %{x}: r = %{y:.3f}<extra>" + label + "</extra>"))
        fig.add_trace(go.Scatter(x=xs, y=[r["exact"] for r in cells], mode="markers", name="exakt", showlegend=False, marker=dict(size=11, symbol="diamond-open", color=colors.get(smooth, C.COLOR_ALL), line=dict(width=2)),
                                 hovertemplate="Fenster %{x}: exakt %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Mittel über … Tage (Fenster)", type="category")
    fig.update_yaxes(title_text="r der Fenstermittel", range=[-0.1, 1.05])
    return _base(fig, height)


def build_select(rows: list, by: str, height: int = 360):
    """E2: r nach Auswahl gegen die Quote der gewählten Einheiten, je ein Linienzug je ρ: gemessen (Linie) und exakt (Rauten); Nulllinie eingezeichnet."""
    fig = go.Figure()
    palette = {0.3: "#9d9d9d", 0.5: C.COLOR_ALL, 0.8: C.COLOR_WITHIN}
    for rho in sorted({r["rho"] for r in rows if r["by"] == by}):
        cells = sorted((r for r in rows if r["by"] == by and r["rho"] == rho), key=lambda r: -r["q"])
        xs = [f"{100 * r['q']:g} %".replace(".", ",") for r in cells]
        color = palette.get(rho, C.COLOR_ALL)
        fig.add_trace(go.Scatter(x=xs, y=[r["mean"] for r in cells], mode="lines+markers", name=f"ρ = {rho:g}".replace(".", ","), line=dict(color=color, width=3),
                                 hovertemplate="oberste %{x}: r = %{y:.3f}<extra>ρ = " + f"{rho:g}" + "</extra>"))
        fig.add_trace(go.Scatter(x=xs, y=[r["exact"] for r in cells], mode="markers", name="exakt", showlegend=False, marker=dict(size=11, symbol="diamond-open", color=color, line=dict(width=2)),
                                 hovertemplate="oberste %{x}: exakt %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=C.COLOR_EXACT, width=1, dash="dot"))
    fig.update_xaxes(title_text="gewählt: oberste … der Einheiten", type="category")
    fig.update_yaxes(title_text="r in der Auswahl", range=[-1, 1])
    return _base(fig, height)


def build_attenuation(rows: list, height: int = 340):
    """E3: wahres r, exakt verdünntes, gemessenes und korrigiertes r je Zuverlässigkeitspaar (Streuung als Balken)."""
    rows = sorted(rows, key=lambda r: -(r["rx"] * r["ry"]))
    labels = [f"{r['rx']:g} / {r['ry']:g}".replace(".", ",") for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["mean"] for r in rows], name="gemessen", marker_color=C.COLOR_ALL, error_y=dict(type="data", array=[r["sd"] for r in rows], visible=True), hovertemplate="%{x}: %{y:.3f}<extra>gemessen</extra>"))
    fig.add_trace(go.Bar(x=labels, y=[r["corrected_mean"] for r in rows], name="korrigiert", marker_color=C.COLOR_WITHIN, error_y=dict(type="data", array=[r["corrected_sd"] for r in rows], visible=True),
                         hovertemplate="%{x}: %{y:.3f}<extra>korrigiert</extra>"))
    fig.add_trace(go.Scatter(x=labels, y=[r["exact"] for r in rows], mode="markers", name="exakt verdünnt", marker=dict(size=12, symbol="diamond-open", color=C.COLOR_EXACT, line=dict(width=2)),
                             hovertemplate="%{x}: exakt %{y:.3f}<extra></extra>"))
    fig.add_hline(y=rows[0]["rho"], line=dict(color=C.COLOR_BETWEEN, width=1.5, dash="dash"), annotation_text="wahres ρ", annotation_position="top right")
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Zuverlässigkeit von X / Y", type="category")
    fig.update_yaxes(title_text="r", rangemode="tozero")
    return _base(fig, height)


def build_levels(rows: list, height: int = 360):
    """E4: je Welt r über alle Standort-Tage, der Standortmittel und innerhalb (Streuung als Balken), exakte Werte als Rauten."""
    labels = [f"{c['g']} Standorte, ρ_b {c['rho_b']:g}, Kopplung {c['gamma']:g}".replace(".", ",") for c in rows]
    fig = go.Figure()
    for key, name, color in (("total", "alle Standort-Tage", C.COLOR_ALL), ("between", "Standortmittel", C.COLOR_BETWEEN), ("within", "innerhalb der Standorte", C.COLOR_WITHIN)):
        fig.add_trace(go.Bar(x=labels, y=[c[key] for c in rows], name=name, marker_color=color, error_y=dict(type="data", array=[c[key + "_sd"] for c in rows], visible=True),
                             hovertemplate="%{x}: %{y:.3f}<extra>" + name + "</extra>"))
        fig.add_trace(go.Scatter(x=labels, y=[c["exact_" + key] for c in rows], mode="markers", name="exakt", showlegend=False, marker=dict(size=11, symbol="diamond-open", color=C.COLOR_EXACT, line=dict(width=2)),
                                 hovertemplate="%{x}: exakt %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text="r", range=[-1, 1])
    return _base(fig, height)
