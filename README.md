# Aggregation und Auswahl: dieselbe Welt, andere Korrelation – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-aggregationsebene-demo.streamlit.app/)** -->

Siebtes Stück der Korrelations-Reihe in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Nachfolger von [pearson-demo](https://github.com/sebastian-hanisch/pearson-demo) (Kante `fix`: r hängt an der Stichprobe und an der Mittelungsebene), neben [rangkorrelation-demo](https://github.com/sebastian-hanisch/rangkorrelation-demo), [abhaengigkeitsmasse-demo](https://github.com/sebastian-hanisch/abhaengigkeitsmasse-demo), [scheinkorrelation-demo](https://github.com/sebastian-hanisch/scheinkorrelation-demo), [korrelationsmatrix-demo](https://github.com/sebastian-hanisch/korrelationsmatrix-demo), [randabhaengigkeit-demo](https://github.com/sebastian-hanisch/randabhaengigkeit-demo) und [korrelierte-merkmale-demo](https://github.com/sebastian-hanisch/korrelierte-merkmale-demo). Es ist die **erste Demo des Portfolios über den Zuschnitt der Daten**: zwei Kennzahlen **X und Y** an **G Standorten über T Tage**; hinter beiden steht eine gemeinsame Welle, dazu kommt tägliches Rauschen. Die Fragen: **Warum steigt r, wenn man Tage zu Wochen mittelt? Was macht die Auswahl „der besten Standorte“ mit r, und warum kann sie das Vorzeichen kehren? Was verdünnt Messrauschen, und was kostet die Korrektur? Welche Ebene beantwortet welche Frage?**

## Warum dieses Problem

Dieselben Zahlen liefern ganz verschiedene Korrelationen, je nachdem, was man mittelt, wen man auswählt und auf welcher Ebene man rechnet. Befunde: bei einer gemeinsamen Welle mit 30 Tagen Gedächtnis und Rauschen 2 hebt die Mittelung r von **0,201** (Tage) über **0,616** (Wochen) auf **0,826** (Monate); hat die Welle kein Gedächtnis, bleibt r bei **jedem** Fenster bei 0,20, aber aus 3 600 Tagen werden 40 Punkte, die Streuung von r steigt auf das Zehnfache und nur noch etwa ein Viertel der Läufe ist „signifikant“ (24,5 % in 200 Läufen, rechnerisch 23 %). Die obersten 5 % von 400 Standorten nach X zeigen bei wahrem ρ = 0,5 nur **0,201** statt 0,500, mit einer Streuung von 0,224. Die **Auswahl nach der Summe** macht aus +0,50 bei den obersten 10 % eine Korrelation von **−0,326**. Und eine Welt liefert gleichzeitig **+0,101** (alle Standort-Tage), **+0,777** (Standortmittel) und **−0,316** (innerhalb der Standorte). Alle vier Mechanismen haben geschlossene Formeln; die Demo zeigt zu jedem Mittelwert den erwarteten Wert daneben (als Verhältnis der Erwartungswerte).

## Befunde und Korrekturen gegenüber dem Plan

Der Einzelplan (`korrelation-planung/STUECK-7-aggregationsebene.md`, nicht in diesem Repo) stützte sich auf Wegwerfskripte mit gleichem Kern, anderer Ziehungsreihenfolge und eigener Seed-Ableitung. Gebaut ist alles mit SplitMix64 und eigenem Kern; die Zahlen weichen leicht ab, die Befunde bleiben, mit diesen Ausnahmen:

- **Mittelung (Glättung 30, σ = 2):** 0,616 / 0,826 für Wochen und Monate (Plan: 0,613 / 0,833 und V1: 0,619 / 0,838); exakt 0,618 / 0,833. Anteil „signifikanter“ Läufe bei weißer Welle und 90-Tage-Fenstern **24,5 %** (Plan: 28 %).
- **Auswahl nach X:** bei 20 gewählten Einheiten Streuung 0,224, **18,8 %** negativ und **14,1 %** signifikant positiv (Plan: 0,222, 19 % und 13 %).
- **Auswahl nach der Summe:** −0,326 (Plan: −0,328) bei den obersten 10 %, exakt −0,327.
- **Ebenen:** der Plan nannte +0,23 / +0,78 / −0,30 mit einer Welt, in der die Tage innerhalb der Standorte direkt mit ρ_w korreliert sind. Gebaut ist **eine** Welt für alle Zuschnitte (Standortniveau, gemeinsame Welle mit Kopplung γ, Rauschen); dort ergeben sich bei ρ_b = 0,8, γ = −0,5, σ = 1 +0,101 / +0,777 / −0,316 (erwartet 0,115 / 0,782 / −0,316).
- **Fenster innerhalb der Standorte:** der Plan mittelte einen einzelnen Standort; in der Hauptansicht mittelt die Zeile „Innerhalb, Mittel über m Tage“ die Abweichungen vom Standortmittel und braucht **mindestens zwei volle Fenster je Standort** (m ≤ T/2); die Mittelung eines einzelnen Standorts steht im Experiment 1.
- **Neu:** der Regler Kopplung γ (Welle in Y gleich, schwächer oder gegenläufig), die Zeilen „ohne Messrauschen“ und „korrigiert“, die Beschränkung der Tage auf 90 bis 720.

## Modell

Standort g, Tag t: x_gt = μx_g + w_gt + σ ε_gt, y_gt = μy_g + γ w_gt + σ η_gt. Die Standortniveaus (μx, μy) sind standardnormal mit Korrelation ρ_b, die Welle w hat Varianz 1 und ist ein gleitendes Mittel über `Gedächtnis` Tage (Autokovarianz max(0, 1 − |h|/Gedächtnis)), ε und η sind unabhängiges Standardnormalrauschen. Erzeuger `agr_algorithm.py`, Zufall aus SplitMix64 in wörtlichen Kopien von `pearson-demo`: `prs_rng.py`, `prs_constants.py`, `prs_scenario.py`, `prs_algorithm.py` (dort auch Pearson und das t-Quantil).

## Methodik

- **Zuschnitte der Hauptansicht** (`agr_evaluation.py`): alle Standort-Tage, dieselben Werte ohne Rauschen, die Korrektur für das Messrauschen (r geteilt durch die Wurzel aus den Zuverlässigkeiten), innerhalb der Standorte (Abweichung vom Standortmittel), Mittel über m Tage dieser Abweichungen, Standortmittel (zwischen den Standorten) und die Auswahl der obersten q % der Standorte nach dem Standortmittel von X oder nach der Summe der standardisierten Werte. Zu jeder Zeile steht der **erwartete** Wert aus der Formel.
- **Exakte Formeln** (Erwartung als Verhältnis der Erwartungswerte): r über alle Standort-Fenster = (ρ_b + γ V_m)/√((1 + V_m + σ²/m)(1 + γ² V_m + σ²/m)) mit der Varianz V_m des Fenstermittels der Welle (für m = 1 alle Standort-Tage, für m = T die Standortmittel); innerhalb mit der Korrektur für das abgezogene Standortmittel; Auswahl nach X nach Thorndike (ρ√u/√(1 − ρ² + ρ² u), u das Varianzverhältnis der gestutzten Normalverteilung); Auswahl nach der Summe (V_S − V_D)/(V_S + V_D); Verdünnung ρ√(rx ry). Die Zerlegung Σ(x − x̄)(y − ȳ) = Σ innerhalb + T · Σ zwischen ist eine Identität.
- **Experimente:** (1) Mittelung und Gedächtnis der Welle (eine Welle ohne Niveau), (2) Auswahl nach X und nach der Summe (normalverteilte Paare), (3) Messrauschen und Korrektur, (4) eine Welt, drei Ebenen.
- **Vorgerechnetes Raster** (`tools/make_grid.py`, `data/aggregation_grid.json`, 2 Sekunden auf 10 Prozessen bei der Erzeugung, rechnerabhängig): Mittelung (24 Zellen: Gedächtnis 1, 7, 30 · Rauschen 1, 2 · Fenster 1, 7, 30, 90, je 4 · 50 Läufe à 3 600 Tage), Auswahl nach X (12 Zellen: ρ = 0,3, 0,5, 0,8 · oberste 100, 50, 20, 5 % von 400 Einheiten, je 4 · 500 Läufe), Auswahl nach der Summe (12 Zellen: ρ · oberste 50, 20, 10, 1 % von 20 000 Paaren, je 4 · 10 Läufe), Messrauschen (5 Zeilen, je 4 · 500 Läufe mit 300 Einheiten), Ebenen (4 Welten, je 4 · 200 Läufe). Vier Seeds je Zelle (101 bis 104). **Live:** eine Welt mit bis zu 60 Standorten und 720 Tagen mit allen Zuschnitten.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (Raster-Werte gegen die JSON-Datei, die erwarteten Werte über dieselben Funktionen wie in der App, die Tabellen Zeile für Zeile in der Rundung der App). In Klammern steht jeweils der **erwartete** Wert (Verhältnis der Erwartungswerte aus der Formel).

**Mittelung über Fenster** (Experiment 1; eine Welle, x = Welle + σ ε, y = Welle + σ η; r der Fenstermittel, Mittel über 200 Läufe à 3 600 Tage; Spalten: Tage, Wochen, Monate, Quartale):

| Gedächtnis der Welle | Rauschen σ | 1 Tag | 7 Tage | 30 Tage | 90 Tage |
|---|---|---|---|---|---|
| 1 | 1 | 0,501 (0,500) | 0,498 (0,500) | 0,501 (0,500) | 0,496 (0,500) |
| 1 | 2 | 0,201 (0,200) | 0,198 (0,200) | 0,206 (0,200) | 0,196 (0,200) |
| 7 | 1 | 0,500 (0,500) | 0,825 (0,825) | 0,866 (0,866) | 0,872 (0,872) |
| 7 | 2 | 0,199 (0,200) | 0,536 (0,541) | 0,615 (0,618) | 0,631 (0,630) |
| 30 | 1 | 0,500 (0,500) | 0,864 (0,866) | 0,952 (0,952) | 0,964 (0,964) |
| 30 | 2 | 0,201 (0,200) | 0,616 (0,618) | 0,826 (0,833) | 0,866 (0,870) |

**Streuung von r und Anteil „signifikanter“ Läufe** (σ = 2; Punkte je Fenster 3 600 / 514 / 120 / 40):

| Gedächtnis | 1 Tag | 7 Tage | 30 Tage | 90 Tage |
|---|---|---|---|---|
| 1 | 0,016 / 100,0 % | 0,039 / 99,5 % | 0,089 / 62,0 % | 0,157 / 24,5 % |
| 30 | 0,023 / 100,0 % | 0,035 / 100,0 % | 0,031 / 100,0 % | 0,041 / 100,0 % |

**Auswahl nach X** (Experiment 2; wahres ρ; r der obersten 100, 50, 20, 5 % von 400 Einheiten, also 400, 200, 80, 20 Einheiten; exakt nach Thorndike):

| ρ | 100 % | 50 % | 20 % | 5 % |
|---|---|---|---|---|
| 0,3 | 0,298 (0,300) | 0,185 (0,186) | 0,144 (0,145) | 0,101 (0,116) |
| 0,5 | 0,500 (0,500) | 0,331 (0,329) | 0,259 (0,261) | 0,201 (0,210) |
| 0,8 | 0,799 (0,800) | 0,626 (0,626) | 0,527 (0,529) | 0,443 (0,444) |

**Auswahl nach der Summe X + Y** (Experiment 2; r der obersten 50, 20, 10, 1 % von 20 000 Paaren; exakt (V_S − V_D)/(V_S + V_D)):

| ρ | 50 % | 20 % | 10 % | 1 % |
|---|---|---|---|---|
| 0,3 | -0,196 (-0,194) | -0,422 (-0,422) | -0,527 (-0,522) | -0,693 (-0,695) |
| 0,5 | 0,046 (0,043) | -0,210 (-0,208) | -0,326 (-0,327) | -0,545 (-0,550) |
| 0,8 | 0,531 (0,532) | 0,325 (0,326) | 0,203 (0,207) | -0,084 (-0,069) |

**Messrauschen und Korrektur** (Experiment 3; wahres ρ = 0,6, 300 Einheiten, 2 000 Läufe je Zeile; Zuverlässigkeit von X und Y; gemessenes r (Streuung); korrigiertes r (Streuung)):

| rx | ry | exakt verdünnt | gemessen (Streuung) | korrigiert (Streuung) |
|---|---|---|---|---|
| 1 | 1 | 0,600 | 0,599 (0,037) | 0,599 (0,037) |
| 0,8 | 0,8 | 0,480 | 0,481 (0,045) | 0,601 (0,056) |
| 0,6 | 0,6 | 0,360 | 0,358 (0,050) | 0,597 (0,084) |
| 0,5 | 0,8 | 0,379 | 0,378 (0,051) | 0,598 (0,080) |
| 0,4 | 0,4 | 0,240 | 0,237 (0,055) | 0,593 (0,137) |

**Eine Welt, drei Ebenen** (Experiment 4; je 800 Läufe, 100 Tage je Standort, Gedächtnis 1, Rauschen 1; r über alle Standort-Tage, der Standortmittel (mit Streuung), innerhalb; Anteil der Läufe mit verschiedenem Vorzeichen von Standortmittel und innerhalb):

| Standorte | ρ_b | Kopplung γ | alle Standort-Tage | Standortmittel | innerhalb | Vorzeichen verschieden |
|---|---|---|---|---|---|---|
| 30 | 0,8 | -0,5 | 0,101 (0,115) | 0,777 (0,782), sd 0,077 | -0,316 (-0,316) | 100,0 % |
| 10 | 0,8 | -0,5 | 0,071 (0,115) | 0,767 (0,782), sd 0,147 | -0,317 (-0,316) | 99,9 % |
| 30 | 0 | 1 | 0,337 (0,333) | 0,009 (0,010), sd 0,185 | 0,499 (0,500) | 48,0 % |
| 30 | 0,8 | 0,5 | 0,495 (0,500) | 0,789 (0,792), sd 0,071 | 0,316 (0,316) | 0,0 % |

| Frage | Befund |
|---|---|
| Hebt die Mittelung r? (Experiment 1) | **Nur, wenn die Welle Gedächtnis hat:** bei 30 Tagen Gedächtnis und σ = 2 steigt r von **0,201** (Tage) auf **0,616** (Wochen), **0,826** (Monate) und **0,866** (Quartale), erwartet 0,200 / 0,618 / 0,833 / 0,870. Ohne Gedächtnis bleibt r bei jedem Fenster bei **0,2** (1/(1 + σ²)). |
| Was kostet die Mittelung? (Experiment 1) | **Punkte:** aus 3 600 Tagen werden 514, 120 und 40 Fenster; bei weißer Welle und σ = 2 steigt die Streuung von r von 0,016 auf 0,157, und der Anteil „signifikanter“ Läufe fällt von 100,0 % auf 24,5 % (200 Läufe; die Rechnung nach Fisher-z gibt 23 %). |
| Was macht die Auswahl der besten Standorte? (Experiment 2) | **Sie schwächt r nach einer Formel:** bei ρ = 0,5 und 400 Einheiten 0,500 → 0,331 → 0,259 → 0,201 für die obersten 100 / 50 / 20 / 5 % (exakt 0,500 / 0,329 / 0,261 / 0,210). Bei nur **20** gewählten Einheiten streut r mit **0,224**, ist in **18,8 %** der Läufe negativ und nur in **14,1 %** signifikant positiv (bei 80: 65,3 %). |
| Was macht die Auswahl nach der Summe? (Experiment 2) | **Sie erzeugt eine negative Korrelation aus einer positiven:** bei ρ = 0,5 +0,046 für die oberen 50 %, **−0,210**, **−0,326** und **−0,545** für 20 / 10 / 1 %; bei ρ = 0,3 schon bei den oberen 50 % −0,196; bei ρ = 0,8 erst bei den obersten 1 % (−0,084 und 85,0 % der Läufe negativ im Raster aus nur 40 Läufen; die Formel gibt −0,069, ein Lauf mit 400 Läufen etwa drei Viertel negativ). |
| Was verdünnt Messrauschen? (Experiment 3) | r = ρ√(rx ry): bei Zuverlässigkeit 0,4 und 0,4 bleibt von ρ = 0,6 nur **0,237** (exakt 0,240). Die Korrektur trifft im Mittel **0,593**, ihre Streuung ist aber **0,137** statt 0,037 ohne Messfehler: sie braucht die Zuverlässigkeit und wird unsicherer. |
| Welche Ebene beantwortet was? (Experiment 4) | **Keine allein:** bei ρ_b = 0,8 und Kopplung −0,5 liefert eine Welt **+0,101** (alle Standort-Tage), **+0,777** (Standortmittel, Streuung 0,077) und **−0,316** (innerhalb), in **100,0 %** der Läufe mit verschiedenem Vorzeichen. Bei nur 10 Standorten streut die Korrelation der Standortmittel mit **0,147**. Ohne Zusammenhang zwischen den Standorten (ρ_b = 0) bleiben Standortmittel 0,009, innerhalb 0,499. |
| Standardwelt der App | 30 Standorte, 360 Tage, ρ_b = 0,5, Rauschen 2, Gedächtnis 30, Kopplung 1: erwartet **0,25** über alle Standort-Tage, **0,53** für die Standortmittel, **0,19** innerhalb, **0,75** ohne Messrauschen; Mittel über 30 Tage innerhalb **0,83**. |
| Handwerte | Zwei Standorte mit x = (0, 2 \| 10, 12), y = (2, 0 \| 12, 10): r gesamt **0,923**, innerhalb **−1**, zwischen **+1**; Kreuzprodukte 96 = −4 + 100. Obere Hälfte einer Normalverteilung: u = 1 − 2/π = **0,3634**, bei ρ = 0,5 also r = **0,3287**. Zuverlässigkeit 0,5 und 0,5 halbiert ρ (0,6 → **0,3**). Bei ρ = 0,5 und den obersten 10 % nach der Summe: exakt **−0,327**. Gesamtkorrelation aus ρ_b = 0,8 und ρ_w = −0,3 bei gleichen Varianzen: **0,25**. |

## Ehrliche Grenzen

- **Die Welt ist additiv und normalverteilt.** Die exakten Formeln gelten für diese Welt (bei der Summenformel gleiche Streuung); echte Kennzahlen weichen ab; ob die Richtung der Effekte bleibt, zeigt die Demo nicht.
- **Die Auswahl folgt einer Regel** (nach X oder nach der Summe). Eine Auswahl, die selbst von Y abhängt, verzerrt anders.
- **Keine Ursachen.** Simpson, Störfaktoren und Collider als Diagramm gehören der Linie *Kausale Diagramme und Adjustierung* (`konfundierung-demo`, `schlechte-kontrollen-demo`); die Summenauswahl ist hier nur das Zahlenbeispiel zur Collider-Mechanik.
- **Die Korrektur des Messrauschens braucht die Zuverlässigkeit.** Sie ist hier bekannt; im Alltag ist sie geschätzt und unsicher.
- **Alle Standorte sind gleich verteilt.** Mehrebenenmodelle mit Zufallseffekten und die Intraklassenkorrelation sind nicht gebaut.
- **Das einzige Gedächtnis ist die Welle.** Trends und Saisonmuster (Stück 4) und räumliche Aggregation (Gebietszuschnitt) fehlen.
- **Erwartet heißt Verhältnis der Erwartungswerte.** Bei wenigen Standorten weicht der Mittelwert der gemessenen r davon merklich ab (10 Standorte: erwartet 0,115 für alle Standort-Tage, gemessen 0,071).
- **Quellen:** keine Literaturangaben; die Formeln sind durch unabhängige Rechenwege (unten) und die Messungen gestützt.

## Verifikation

- **Orakel** (`tests/test_oracle_aggregationsebene.py`): das Varianzverhältnis der gestutzten Normalverteilung gegen **`scipy.stats.truncnorm`**; die Thorndike- und die Summenformel gegen **Simulationen mit einer Million Paaren** und die erste auch gegen **numerische Integration** (`scipy.integrate.quad`); die Varianz des Fenstermittels der Welle gegen die volle **Toeplitz-Kovarianzmatrix** (`scipy.linalg.toeplitz`); die erwarteten Korrelationen (Fenster, gesamt, zwischen, innerhalb, innerhalb mit Fenstern) gegen **Simulationen der Welt**; die Zerlegung gegen die **Schleife über Standorte**; Pearson gegen `numpy.corrcoef` und `scipy.stats.pearsonr`; die Verdünnung gegen die Simulation und die Korrektur auf Unverzerrtheit.
- **Kern** (`tests/test_algorithm.py`, `tests/test_edge_cases.py`): Handrechnungen (Mini-Instanz in `conftest.py`: zwei Standorte mit 96 = −4 + 100, obere Hälfte, Fenster ohne Gedächtnis, Verdünnung), Reglergrenzen, **inklusive Grenzen** (zwei volle Fenster je Standort, mindestens drei gewählte Standorte, Quote 1), konstante Werte, kleinste Welt.
- **Eigenschaften** (`tests/test_eigenschaften_aggregationsebene.py`): Invarianz gegen Verschieben und Strecken, Vorzeichenwechsel bei Spiegelung, Symmetrie in X und Y, Reihenfolge der Standorte und Tage, Zerlegung in vielen Welten, Auswahl invariant gegen monotone Umformung, Monotonie der exakten Formeln.
- **Raster und Aussagen** (`tests/test_grid.py`, `tests/test_claims.py`): Probe des Erzeugers gegen die gespeicherte, jede Zahl dieser README gegen das Raster oder die Live-Funktion, die Tabellen in der Rundung der App.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an seinen Grenzen), alle Presets, Adresszeile, jedes Auswahlfeld der Experimente, Layout, Footer, Laufzeit-Importe.
- **Mutationsprüfung:** `tools/mutate_demo.py` des Website-Repos mutiert den Kern `agr_algorithm` (die wörtlichen Kopien `prs_*` sind ausgenommen, sie werden in `pearson-demo` geprüft; Auswertung, Verdikte und Formen prüfen die übrigen Tests): 129 Stellen, alle gesichert; 124 durch eigene Tests gefangen, 1 durch eine wortgleiche Zeile in einer Nachbar-Demo und **4 als gleichwertig festgehalten**: die Summe Σ(X − x̄)(Y − ȳ) ändert sich nicht, wenn man in einem der beiden Faktoren das Vorzeichen des Mittelwerts umkehrt, weil die Abweichungen vom eigenen Mittel zu null summieren (in der Gesamtsumme und in der Summe der Standortmittel). Der erste Lauf fand 13 Überlebende; 9 davon waren echte Lücken und führten zu Tests mit Fehlermeldungen (Wertebereich der Fenster, der Quote und der Zuverlässigkeit, `smooth` unter 1), zu einer Probe mit anderen Streuungen der Tage für die erwartete Gesamtkorrelation und zu einem Test der Standardisierung.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `agr_algorithm.py` | Kern: Welt, Fenstermittel, Auswahl, Zerlegung, Verdünnung und Korrektur, alle exakten Formeln |
| `agr_constants.py` | Regler-Grenzen, Auswahlarten, Raster der Messreihen, Farben, Presets |
| `agr_evaluation.py` | Einzelwelt mit ihren Zeilen, Zellen des Rasters, Verdikte |
| `agr_layout.py` | Fester Kopf (Stylesheet) |
| `agr_pdf_export.py` | PDF-Export der Hauptansicht |
| `agr_presets.py` | Permalink, Presets (Schnellstart mit Hervorhebung), Zufalls-Seed-Button |
| `agr_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `app.py` | Streamlit-App |
| `data/aggregation_grid.json` | vorgerechnetes Raster (erzeugt von `tools/make_grid.py`) |
| `prs_*.py` | wörtliche Kopien aus `pearson-demo` (SplitMix64, Szenario, Pearson-Kern, Konstanten) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy für die Orakel |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Orakel, Kern, Eigenschaften, Raster, Aussagen, Verdikte, Oberfläche, Layout, Footer, Laufzeit-Importe, Kopie-Test |
| `tools/make_grid.py` | rechnet das Raster vor |

## Bewusst nicht umgesetzt

- **Simpson-Paradox und Collider als Diagramm** (Ursachen): Linie *Kausale Diagramme und Adjustierung*; hier nur die Summenauswahl als Formel.
- **Mehrebenenmodelle mit Zufallseffekten und die Intraklassenkorrelation** als eigenes Maß.
- **Räumliche Aggregation** (Gebietszuschnitt, MAUP) in zwei Dimensionen.
- **Auswahl nach Y oder nach dem Ergebnis** und Auswahl mit Zufall statt nach einer Regel.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Das Raster entsteht neu mit `python tools/make_grid.py` (bei der letzten Erzeugung 2 Sekunden auf zehn Prozessen; je nach Rechner anders); `tests/test_grid.py` rechnet eine feste Probe nach und schlägt an, wenn Erzeuger oder Statistik das Raster überholt haben.

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
