from pathlib import Path
from typing import Sequence

import pandas as pd

def load_features(feature_paths: Sequence[str], features: set[str]) -> pd.DataFrame:
    if not feature_paths:
        raise ValueError(
            'no feature tables supplied; pass --features with the per-atom '
            'parquet(s) of the upstream analyses'
        )

    tables = []
    for raw in feature_paths:
        path = Path(raw)
        if not path.is_file():
            raise ValueError(f'feature table not found: {path}')
        tables.append(pd.read_parquet(path).set_index('atom_index'))

    columns = {
        name: table[name] for table in tables
        for name in table.columns.intersection(features)
    }

    missing = sorted(features - set(columns))
    if missing:
        available = sorted({str(name) for table in tables for name in table.columns})
        raise ValueError(
            f'feature(s) {", ".join(missing)} are not in any supplied table. '
            f'Available columns: {", ".join(available)}'
        )

    return tables[0][['id', 'x', 'y', 'z']].assign(**columns).sort_index().reset_index()
