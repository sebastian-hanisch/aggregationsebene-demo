"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def two_sites():
    """Zwei Standorte mit je zwei Tagen: x = (0, 2 | 10, 12), y = (2, 0 | 12, 10). Standortmittel (1, 1) und (11, 11); gesamt x̄ = ȳ = 6.
    Σ(x − 6)(y − 6) = 24 + 24 + 24 + 24 = 96, Σ(x − 6)² = 104, also r gesamt = 96/104 = 0,923. Innerhalb: Abweichungen (−1, 1) und (1, −1) je Standort, Σ = −2 − 2 = −4, Σ dx² = 4, r = −1.
    Zwischen: Mittel x = (1, 11), y = (1, 11): r = +1, Kreuzprodukt T · Σ = 2 · (25 + 25) = 100; Zerlegung: 96 = −4 + 100."""
    return np.array([[0.0, 2.0], [10.0, 12.0]]), np.array([[2.0, 0.0], [12.0, 10.0]])
