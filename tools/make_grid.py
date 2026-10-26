"""Rechnet das Raster der Messreihen vor (E1 bis E4) und schreibt data/aggregation_grid.json.

    python tools/make_grid.py            # volles Raster, mehrere Prozesse
    python tools/make_grid.py --quick    # kleines Probe-Raster für Tests (nicht in data/aggregation_grid.json)

Das Raster ist deterministisch (SplitMix64, feste Seeds); tests/test_grid.py rechnet eine feste Probe nach und vergleicht sie mit der Datei."""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agr_constants as C  # noqa: E402
import agr_evaluation as ev  # noqa: E402

JOBS = {"window": ev.window_cell, "select": ev.select_cell, "attenuation": ev.attenuation_cell, "level": ev.level_cell}


def _run(task):
    kind, args = task
    return kind, JOBS[kind](*args)


def main() -> None:
    quick = "--quick" in sys.argv
    tasks = [("select", ("summe", rho, q, C.SUM_N, C.SUM_RUNS)) for rho in C.SELECT_RHOS for q in C.SUM_QS]                  # die teuren Zellen zuerst
    tasks += [("window", (smooth, sigma, m)) for smooth in C.WINDOW_SMOOTHS for sigma in C.WINDOW_SIGMAS for m in C.WINDOWS]
    tasks += [("select", ("x", rho, q, C.SELECT_N, C.SELECT_RUNS)) for rho in C.SELECT_RHOS for q in C.SELECT_QS]
    tasks += [("attenuation", rels) for rels in C.ATT_RELS]
    tasks += [("level", case) for case in C.LEVEL_CASES]
    if quick:
        tasks = [("window", (7, 1.0, 7, 2, (7,), 600)), ("select", ("x", 0.5, 0.2, 100, 3, (7,))), ("attenuation", (0.6, 0.8, 3, (7,), 100)), ("level", (8, 40, 0.8, 1.0, 1, -0.5, 3, (7,)))]
    t0 = time.time()
    out = {k: [] for k in JOBS}
    with ProcessPoolExecutor(max_workers=10) as pool:
        for kind, row in pool.map(_run, tasks, chunksize=1):
            out[kind].append(row)
    grid = {"meta": {"seeds": list(C.GRID_SEEDS), "seconds": round(time.time() - t0)}, **out, "probe": ev.grid_probe()}
    dest = ROOT / "data" / ("aggregation_grid_quick.json" if quick else "aggregation_grid.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dest} geschrieben: " + ", ".join(f"{len(v)} {k}" for k, v in out.items()) + f", {grid['meta']['seconds']} s")


if __name__ == "__main__":
    main()
