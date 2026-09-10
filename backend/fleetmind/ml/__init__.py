"""FleetMind ML subpackage: per-depot bearing fault classification (1D-CNN).

Phases:
    train.py      train one local CNN per depot (Phase 2)
    evaluate.py   evaluate a depot model on held-out test data
    model.py      BearingCNN (the architecture)
    dataset.py    Postgres-backed torch Dataset for signal_windows
"""
