"""Aggregation und Auswahl: dieselbe Welt, andere Korrelation - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Siebtes Stück der Korrelations-Reihe. Zwei Kennzahlen an G Standorten über T Tage: eine Welt, mehrere Zuschnitte. Mittelung über Fenster, Auswahl der besten Standorte (nach X und nach der Summe),
Messrauschen und die Ebene (alle Standort-Tage, Standortmittel, innerhalb der Standorte), jeweils mit der exakt erwarteten Korrelation daneben.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import agr_constants as C
import agr_evaluation as ev
from agr_layout import dunkles_design, kopf_css
from agr_pdf_export import build_pdf
from agr_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from agr_visualization import (
    ROW_COLORS,
    build_attenuation,
    build_levels,
    build_rows_bar,
    build_scatter,
    build_select,
    build_window,
)

st.set_page_config(page_title="Aggregation und Auswahl – Sebastian Hanisch", layout="wide")


def de(v: float, digits: int = 2) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{v:.{digits}f}".replace(".", ",")


def n_text(v: float) -> str:
    return f"{v:g}".replace(".", ",")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _grid():
    return ev.load_grid()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in agr_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧮 Aggregation und Auswahl: dieselbe Welt, andere Korrelation")
st.markdown(
    """
**Siebtes Stück der Korrelations-Reihe** und **die erste Demo des Portfolios über den Zuschnitt der Daten**. Zwei Kennzahlen **X und Y** werden an **G Standorten über T Tage** gemessen; hinter beiden
steht eine gemeinsame Welle, dazu kommt tägliches Rauschen. Dieselben Zahlen liefern ganz verschiedene Korrelationen, je nachdem, **was man mittelt, wen man auswählt und auf welcher Ebene man rechnet**.
Die Fragen: **Warum steigt r, wenn man Tage zu Wochen mittelt? Was macht die Auswahl „der besten Standorte“ mit r? Was verdünnt Messrauschen? Und welche Ebene beantwortet welche Frage?**

Wer die Idee zuerst ohne Fachbegriffe verstehen will, öffnet **💬 In einfachen Worten**. Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**. Ursachen, Störfaktoren und Collider als Diagramm stehen in der Linie
*Kausale Diagramme und Adjustierung* (`konfundierung-demo`, `schlechte-kontrollen-demo`); hier geht es nur um den **Zuschnitt**, mit exakten Formeln statt Diagrammen.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/aggregationsebene-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Stellen Sie sich zwei Kennzahlen vor, etwa Sendungen und Personal, gemessen an vielen Standorten über viele Tage. Hinter beiden steht eine gemeinsame Schwankung, dazu kommt tägliches Rauschen. Die Demo zeigt: Dieselben Zahlen liefern ganz verschiedene **Korrelationen**, je nachdem, **was man mittelt, wen man auswählt und auf welcher Ebene man rechnet**.

Tage zu Wochen zu mitteln hebt die Korrelation nur, wenn die gemeinsame Schwankung über mehrere Tage anhält. Wer nur die „besten“ Standorte betrachtet, sieht eine schwächere, bei Auswahl nach der Summe sogar eine umgekehrte. Zwischen den Standorten und innerhalb der Standorte kann dieselbe Welt verschiedene Vorzeichen haben. Zu jedem Wert steht daneben der Wert, den die Formel erwartet.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/aggregationsebene-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        r"""
| Begriff | Was er hier bedeutet |
|---|---|
| **Welt** | Jeder Standort hat ein Niveau (X und Y hängen dort mit ρ_b zusammen) und eine **Welle**, die X und Y gemeinsam bewegt (Kopplung γ, Gedächtnis in Tagen), plus tägliches **Rauschen** σ. |
| **Mittelung** | Statt einzelner Tage werden Mittel über Fenster von 7, 30 oder 90 Tagen korreliert. Das Rauschen mittelt sich weg, die Welle bleibt: r steigt, **wenn die Welle Gedächtnis hat**. |
| **Auswahl** | Nur die obersten q % der Standorte nach X oder nach der Summe aus X und Y werden betrachtet (Bereichseinschränkung). Das schwächt r nach einer festen Formel; nach der Summe kann das Vorzeichen kippen. |
| **Messrauschen** | Rauschen verdünnt r um die Wurzel aus den Zuverlässigkeiten; die Korrektur ist näherungsweise unverzerrt, braucht aber die Zuverlässigkeit und streut stärker. |
| **Ebene** | **Alle Standort-Tage**, **Standortmittel** (zwischen den Standorten) und **innerhalb** der Standorte (Abweichung vom Standortmittel) beantworten drei verschiedene Fragen. |
| **Erwartet (exakt)** | Zu jeder Zeile steht der Wert, den die Formel für diese Welt erwartet; der gemessene Wert schwankt um ihn. |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Welt**")
    g = st.slider("Standorte G", *bounds("g_slider"), key="g_slider", help="Zahl der Standorte. Die Korrelation der Standortmittel beruht nur auf G Punkten.")
    t = st.slider("Tage T je Standort", *bounds("t_slider"), key="t_slider", help="Zahl der Tage je Standort.")
    rho_b = st.slider("Korrelation ρ_b der Standortniveaus", *bounds("rho_b_slider"), step=0.1, key="rho_b_slider", format="%.1f", help="Wie stark X und Y zwischen den Standorten zusammenhängen.")
    sigma = st.slider("Rauschen σ (Messfehler)", *bounds("sigma_slider"), step=0.5, key="sigma_slider", format="%.1f", help="Streuung des täglichen Rauschens in X und Y, bezogen auf die Welle (Streuung 1).")
    smooth = st.select_slider("Gedächtnis der Welle (Tage)", options=C.SMOOTHS, key="smooth_select", help="1 = jeder Tag neu, 30 = die Welle ändert sich über Wochen.")
    gamma = st.slider("Kopplung γ der Welle an Y", *bounds("gamma_slider"), step=0.25, key="gamma_slider", format="%.2f", help="1: Welle wirkt in X und Y gleich. Negativ: Y läuft gegen X.")
    st.markdown("**Zuschnitt**")
    window = st.select_slider("Mittel über … Tage", options=C.WINDOWS, key="window_select", help="Fenster der Mittelung innerhalb der Standorte (1 = Tageswerte).")
    select = st.selectbox("Auswahl der Standorte", list(C.SELECTIONS), format_func=C.SELECTIONS.get, key="select_select", help="Nur die obersten q % der Standorte nach dem Standortmittel von X oder nach der Summe von X und Y betrachten.")
    q = st.slider("Anteil q der gewählten Standorte (%)", *bounds("q_slider"), step=5, key="q_slider", help="Wird nur bei einer Auswahl verwendet; mindestens 3 Standorte.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Welt ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"g_slider": int(g), "t_slider": int(t), "rho_b_slider": float(rho_b), "sigma_slider": float(sigma), "smooth_select": int(smooth), "gamma_slider": float(gamma),
                   "window_select": int(window), "select_select": select, "q_slider": int(q), "seed_input": int(seed)})

settings = ev.Settings(int(g), int(t), float(rho_b), float(sigma), int(smooth), float(gamma), int(window), select, int(q), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
by_key = {r.key: r for r in a.rows}
cols = st.columns(3)
cols[0].metric("Alle Standort-Tage", de(by_key["total"].r), help=f"r über alle {by_key['total'].points} Werte; erwartet {de(by_key['total'].exact)}.")
cols[1].metric("Standortmittel (zwischen)", de(by_key["between"].r), help=f"r über die {by_key['between'].points} Standortmittel; erwartet {de(by_key['between'].exact)}.")
cols[2].metric("Innerhalb der Standorte", de(by_key["within"].r), help=f"r der Abweichungen vom Standortmittel; erwartet {de(by_key['within'].exact)}.")
cols = st.columns(3)
cols[0].metric(f"Mittel über {settings.window} Tage" if "window" in by_key else "Fenster", de(by_key["window"].r) if "window" in by_key else "–", help="Nur bei einem Fenster über 1 Tag.")
cols[1].metric("Gewählte Standorte" if "selected" in by_key else "Auswahl", de(by_key["selected"].r) if "selected" in by_key else "–", help="Nur bei einer Auswahl.")
cols[2].metric("Ohne Messrauschen", de(by_key["true"].r), help=f"r der wahren Werte; erwartet {de(by_key['true'].exact)}.")
st.plotly_chart(build_rows_bar(a.rows), width="stretch", key="main_bars")
st.caption("Alle Zuschnitte derselben Welt: gemessen (Balken) und erwartet (Rauten). Die Balken weichen von den Rauten durch den Zufall dieser einen Welt und, bei wenigen Standorten, durch die Näherung (Verhältnis der Erwartungswerte) ab.")
names = {r.key: r.name for r in a.rows if r.key in a.data}
view = st.selectbox("Zuschnitt für das Streudiagramm", list(names), format_func=names.get, key="view_select")
vx, vy = a.data[view]
st.plotly_chart(build_scatter(vx, vy, by_key[view].r, names[view], ROW_COLORS.get(view, C.COLOR_ALL), selected=a.selected if view == "between" else None), width="stretch", key="main_scatter")
kind, text = ev.verdict_main(a)
getattr(st, kind)(text)
for verdict in (ev.verdict_window, ev.verdict_selection, ev.verdict_noise):
    v = verdict(a)
    if v is not None:
        getattr(st, v[0])(v[1])
if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = (settings, build_pdf(a))
pdf = st.session_state.get("pdf_bytes")
if pdf and pdf[0] == settings:
    st.download_button("⬇️ PDF herunterladen", pdf[1], file_name="aggregationsebene-demo.pdf", mime="application/pdf", key="pdf_download")

st.markdown("## ❓ Kernfrage: Welche Korrelation ist „die“ Korrelation?")
st.markdown("**Keine einzelne:** jeder Zuschnitt beantwortet eine andere Frage (Tage verstehen, Standorte vergleichen, die Besten betrachten), und r hängt an dieser Frage und an der Zahl der Punkte (Experimente unten).")

st.markdown("---")

grid = _grid()
if grid is None:
    st.warning("Das vorgerechnete Raster (data/aggregation_grid.json) fehlt; es entsteht mit `python tools/make_grid.py`. Die Experimente zeigen es.")

# --- E1 Mittelung ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Mittelung und Gedächtnis der Welle")
st.caption(f"Eine Welle (ohne Standortniveau), x = Welle + σ·Rauschen, y = Welle + σ·Rauschen; {C.WINDOW_T:,} Tage".replace(",", " ") + f", {4 * C.WINDOW_RUNS} Läufe je Zelle. Linien: gemessen (Streuung als Balken), Rauten: exakt.")
if grid is not None:
    e1c1, e1c2 = st.columns(2)
    e1_sigma = e1c1.selectbox("Rauschen σ", list(C.WINDOW_SIGMAS), index=1, key="e1_sigma", format_func=n_text)
    e1_smooth = e1c2.selectbox("Gedächtnis der Welle für das Verdikt (Tage)", list(C.WINDOW_SMOOTHS), index=2, key="e1_smooth")
    st.plotly_chart(build_window(grid["window"], e1_sigma), width="stretch", key="window_chart")
    kind, text = ev.verdict_window_grid(grid["window"], e1_smooth, e1_sigma)
    getattr(st, kind)(text)
    st.caption("Mittelung hebt r nur, wenn der gemeinsame Teil Gedächtnis hat. Ohne Gedächtnis bleibt r bei 1/(1 + σ²) für jedes Fenster; dann kostet die Mittelung nur Punkte. Aus 3 600 Tagen werden bei 90-Tage-Fenstern 40 Punkte.")

st.markdown("---")

# --- E2 Auswahl ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Auswahl nach X und nach der Summe")
st.caption(f"Normalverteilte Paare mit wahrem ρ; Auswahl der obersten Quote. Nach X: {C.SELECT_N} Einheiten, {4 * C.SELECT_RUNS} Läufe je Zelle; nach der Summe: {C.SUM_N:,} Einheiten, {4 * C.SUM_RUNS} Läufe. Rauten: exakte Formel.".replace(",", " "))
if grid is not None:
    e2c1, e2c2 = st.columns(2)
    e2_by = e2c1.selectbox("Auswahl", ["x", "summe"], format_func=lambda v: "nach X" if v == "x" else "nach der Summe X + Y", key="e2_by")
    e2_rho = e2c2.selectbox("wahres ρ", list(C.SELECT_RHOS), index=1, key="e2_rho", format_func=n_text)
    st.plotly_chart(build_select(grid["select"], e2_by), width="stretch", key="select_chart")
    kind, text = ev.verdict_select_grid(grid["select"], e2_by, e2_rho)
    getattr(st, kind)(text)
    st.caption("Die Auswahl nach X schwächt r nach der Formel von Thorndike. Mit wenigen gewählten Einheiten ist r nicht nur schwach, sondern unsicher bis zum Vorzeichen. "
               "Die Auswahl nach der Summe stutzt nur die Summe, nicht die Differenz: daraus entsteht eine negative Korrelation (der Collider-Mechanismus als Formel).")

st.markdown("---")

# --- E3 Messfehler ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Messrauschen verdünnt, die Korrektur streut")
st.caption(f"Wahres ρ = {n_text(C.ATT_RHO)}; beobachtet mit Zuverlässigkeit rx, ry (Anteil der wahren Varianz); {C.ATT_N} Standorte, {4 * C.ATT_RUNS} Läufe je Zeile.")
if grid is not None:
    st.plotly_chart(build_attenuation(grid["attenuation"]), width="stretch", key="attenuation_chart")
    kind, text = ev.verdict_attenuation_grid(grid["attenuation"])
    getattr(st, kind)(text)
    st.caption("Messrauschen verkleinert r um die Wurzel aus dem Produkt der Zuverlässigkeiten. Dividiert man durch diese Wurzel, trifft man das wahre ρ im Mittel, aber nur mit bekannter Zuverlässigkeit und mit größerer Streuung.")

st.markdown("---")

# --- E4 Ebenen --------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Eine Welt, drei Ebenen")
st.caption(f"Vier Welten mit je {4 * C.LEVEL_RUNS} Läufen: r über alle Standort-Tage, der Standortmittel und innerhalb der Standorte (Abweichung vom Standortmittel); Rauten: erwartet.")
if grid is not None:
    st.plotly_chart(build_levels(grid["level"]), width="stretch", key="levels_chart")
    labels = [f"{c['g']} Standorte, ρ_b {n_text(c['rho_b'])}, Kopplung {n_text(c['gamma'])}" for c in grid["level"]]
    e4_case = st.selectbox("Welt für das Verdikt", list(range(len(labels))), format_func=lambda i: labels[i], key="e4_case")
    kind, text = ev.verdict_levels_grid(grid["level"], e4_case)
    getattr(st, kind)(text)
    st.caption("Aus einem positiven Zusammenhang zwischen den Standorten folgt nichts für die Tage innerhalb eines Standorts (ökologischer Fehlschluss), und umgekehrt. Mit wenigen Standorten streut die Korrelation der Standortmittel stark.")

st.markdown("---")

# --- Grenzen ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    r"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Welt ist additiv und normalverteilt** | die exakten Formeln gelten für diese Welt (Normalverteilung, gleiche Streuung bei der Summenformel); echte Kennzahlen weichen ab; ob die Richtung der Effekte bleibt, zeigt die Demo nicht | – |
| **Die Auswahl folgt einer Regel** | hier wird nach X oder nach der Summe gewählt; eine Auswahl, die selbst von Y abhängt (nach Y, nach dem Ergebnis), verzerrt anders | Linie *Kausale Diagramme und Adjustierung* |
| **Ursachen** | die Demo zeigt keine Wirkung: Simpson, Störfaktoren und Collider als Diagramm gehören der Kausal-Linie; die Summenauswahl ist nur ihr Zahlenbeispiel | `konfundierung-demo`, `schlechte-kontrollen-demo` |
| **Die Zuverlässigkeit ist bekannt** | die Korrektur des Messrauschens braucht sie; im Alltag ist sie geschätzt und unsicher | – |
| **Alle Standorte sind gleich verteilt** | echte Standorte unterscheiden sich in Streuung und Gedächtnis; Mehrebenenmodelle mit Zufallseffekten sind nicht gebaut | – |
| **Unabhängige Tage außerhalb der Welle** | die Welle ist das einzige Gedächtnis; Trends und Saisonmuster würden die Fensterwerte weiter verschieben (Stück 4) | Stück 4 |
| **Räumliche Aggregation** | die Mittelung läuft nur über die Zeit, nicht über Gebiete (MAUP) | – |
"""
)
st.caption("Alle Daten sind simuliert; die Welt ist eine Modellwahl, kein Befund über echte Standorte.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/aggregationsebene-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $G$ Standorte mit je $T$ Tagen und zwei Kennzahlen $X$ und $Y$. Gesucht ist die Korrelation $r$ je nach Zuschnitt: alle Standort-Tage, Fenstermittel, Standortmittel, innerhalb der Standorte, nach einer Auswahl und bei Messrauschen.

| Symbol | Bedeutung |
|---|---|
| $x_{gt}$, $y_{gt}$ | Kennzahlen $X$ und $Y$ an Standort $g$ und Tag $t$ |
| $\mu^x_g$, $\mu^y_g$ | Niveau des Standorts $g$ in $X$ und $Y$ (Standardnormal) |
| $\rho_b$ | Korrelation der Standortniveaus (zwischen den Standorten) |
| $w_{gt}$ | gemeinsame Welle (Varianz 1); das Gedächtnis $s$ sind die Tage, über die sie anhält |
| $\gamma$ | Kopplung der Welle an $Y$ (1: gleich wie in $X$, negativ: gegenläufig) |
| $\sigma\,\varepsilon_{gt}$, $\sigma\,\eta_{gt}$ | tägliches Rauschen in $X$ und $Y$ (Standardnormal, Stärke $\sigma$) |
| $m$ | Zahl der Tage im Fenster |
| $V_m$ | Varianz des Fenstermittels der Welle |
| $q$ | Anteil der gewählten Einheiten |
| $\rho$ | wahre Korrelation vor der Auswahl bzw. vor dem Messrauschen |
| $r_x$, $r_y$ | Zuverlässigkeiten von $X$ und $Y$ (Anteil der wahren Varianz) |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/aggregationsebene-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Modell.** An Standort $g$ und Tag $t$ gilt

$$ x_{gt} = \mu^x_g + w_{gt} + \sigma\,\varepsilon_{gt}, \qquad y_{gt} = \mu^y_g + \gamma\, w_{gt} + \sigma\,\eta_{gt} $$

mit Standortniveaus $(\mu^x_g, \mu^y_g)$ (Standardnormal, Korrelation $\rho_b$), einer Welle $w$ (Varianz 1, Autokovarianz $\max(0, 1 - |h|/s)$ bei Gedächtnis $s$) und Standardnormalrauschen $\varepsilon, \eta$.

**Fenstermittel der Welle.**

$$ V_m = \frac{1}{m^2}\Big(m + 2\sum_{h=1}^{m-1}(m-h)\max\big(0,\,1-\tfrac{h}{s}\big)\Big) $$

**Erwartete Korrelation** über alle Standort-Fenster (für $m = 1$ alle Standort-Tage, für $m = T$ die Standortmittel):

$$ r_m = \frac{\rho_b + \gamma V_m}{\sqrt{\big(1 + V_m + \sigma^2/m\big)\big(1 + \gamma^2 V_m + \sigma^2/m\big)}} $$

Ohne Standortniveau und mit $\gamma = 1$ ist $r_m = V_m/(V_m + \sigma^2/m)$; ohne Gedächtnis ($V_m = 1/m$) ist das $1/(1+\sigma^2)$ für jedes $m$.

**Auswahl nach $X$** (obere Quote $q$, wahres $\rho$, Thorndike): mit $c = \Phi^{-1}(1-q)$, $\lambda = \varphi(c)/q$ und $u = 1 + \lambda c - \lambda^2$

$$ r_{\text{Auswahl}} = \frac{\rho\sqrt{u}}{\sqrt{1 - \rho^2 + \rho^2 u}} $$

**Auswahl nach der Summe** (gleiche Streuung): $D = X - Y$ bleibt unberührt, $S = X + Y$ wird gestutzt, $V_S = (2+2\rho)\,u$, $V_D = 2 - 2\rho$:

$$ r_{\text{Summe}} = \frac{V_S - V_D}{V_S + V_D} $$

**Messrauschen** mit den Zuverlässigkeiten $r_x, r_y$ (Anteil der wahren Varianz) und die Korrektur:

$$ r_{\text{beob}} = \rho\sqrt{r_x r_y}, \qquad \hat\rho = \frac{r_{\text{beob}}}{\sqrt{r_x r_y}} $$

**Zerlegung** der Kreuzproduktsumme (jeder Standort mit $T$ Tagen, Mittel $\bar x_g$ je Standort, $\bar x$ gesamt):

$$ \sum_{g,t}(x_{gt}-\bar x)(y_{gt}-\bar y) = \sum_{g,t}(x_{gt}-\bar x_g)(y_{gt}-\bar y_g) + T\sum_g(\bar x_g-\bar x)(\bar y_g-\bar y) $$

**Quellen.** Der Name Thorndike (Bereichseinschränkung) wird hier ohne Literaturangabe genannt; die Formeln sind durch unabhängige Rechenwege (Tests gegen scipy, Simulationen und Schleifenfassungen) und die Messungen der Experimente gestützt.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
