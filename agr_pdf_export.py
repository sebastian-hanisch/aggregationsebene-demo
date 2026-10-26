"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Zuschnitte mit r und erwartetem Wert, Verdikte der Hauptansicht. Kein Gedankenstrich und kein Euro-Zeichen (Kernschriften)."""

from fpdf import FPDF

import agr_constants as C
import agr_evaluation as ev


def _text(value: str) -> str:
    """Zeichen, die die Kernschriften nicht kennen, ersetzen (Gedankenstrich, Pfeile, Fettdruck-Markierung, griechische und mathematische Zeichen)."""
    table = {"–": "-", "—": "-", "→": "->", "**": "", "€": "EUR", "·": "*", "≤": "<=", "≥": ">=", "−": "-", "×": "x", "„": '"', "“": '"', "ρ": "rho", "σ": "sigma", "γ": "gamma", "²": "^2", "√": "sqrt ", "…": "...", "π": "pi"}
    for old, new in table.items():
        value = value.replace(old, new)
    return value.encode("latin-1", "replace").decode("latin-1")


def _para(pdf: FPDF, height: float, text: str) -> None:
    """Absatz über die volle Breite; danach steht der Cursor wieder am linken Rand."""
    pdf.multi_cell(0, height, _text(text), new_x="LMARGIN", new_y="NEXT")


def build_pdf(a: ev.Analysis) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _text("Aggregation und Auswahl: dieselbe Welt, andere Korrelation"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    _para(pdf, 5.5, f"Standorte: {s.g}   Tage: {s.t}   rho_b: {s.rho_b:g}   Rauschen sigma: {s.sigma:g}   Gedaechtnis der Welle: {s.smooth} Tage   Kopplung gamma: {s.gamma:g}   Fenster: {s.window} Tage   "
                      f"Auswahl: {C.SELECTIONS[s.select]}{'' if s.select == 'keine' else f' ({s.q} %)'}   Zufalls-Seed: {s.seed}")
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Zuschnitte", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for r in a.rows:
        exact = "-" if r.exact is None else f"{r.exact:.3f}"
        _para(pdf, 5.5, f"{r.name}: r = {r.r:.3f} (erwartet {exact}), {r.points} Punkte")
    for title, verdict in (("Ergebnis", ev.verdict_main), ("Mittelung", ev.verdict_window), ("Auswahl", ev.verdict_selection), ("Messrauschen", ev.verdict_noise)):
        v = verdict(a)
        if v is None:
            continue
        pdf.ln(1)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, _text(title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        _para(pdf, 5.5, v[1])
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    _para(pdf, 4.5, "Simulierte Daten mit bekannter Welt. Sebastian Hanisch - Operations Research und Machine Learning.")
    return bytes(pdf.output())
