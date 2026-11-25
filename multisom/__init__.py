"""MultiSOM — multilayer Self-Organizing-Map structure identification.

Public API:
    run_multisom(df, layers, log, mapping='linear', seed=1982, capture_model=False)
        -> (cluster_id, layer_columns[, model])
    apply_multisom(df, model, log) -> (cluster_id, layer_columns)
"""

from .pipeline import run_multisom, apply_multisom, normalize_layers, normalize_data

__all__ = ['run_multisom', 'apply_multisom', 'normalize_layers', 'normalize_data']
