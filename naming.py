"""
Agent naming shared across the analysis and figure scripts.

The results JSON stores agents under their stored keys (the keys of
experiment.AGENTS). Figures and printouts use the display labels of the form
OM-X so the names read consistently and match the thesis.
"""

# Stored keys in plotting order. Must match experiment.AGENTS keys.
STORED_AGENTS = ["Baseline", "Det-Only", "Sel-Only", "Full-OM"]

# stored key -> display label
DISPLAY = {
    "Baseline":  "Baseline",
    "Det-Only":  "OM-Det",
    "Sel-Only":  "OM-Sel",
    "Full-OM":   "OM-Full",
}
# display label -> stored key
STORED = {v: k for k, v in DISPLAY.items()}

# display label -> matplotlib color
COLORS = {
    "Baseline":  "C0",
    "OM-Det":    "C1",
    "OM-Sel":    "C2",
    "OM-Full":   "C3",
}

# display labels in plotting order
DISPLAY_AGENTS = [DISPLAY[a] for a in STORED_AGENTS]


def disp(stored_name):
    """Stored key -> display label (identity for unknown names)."""
    return DISPLAY.get(stored_name, stored_name)
