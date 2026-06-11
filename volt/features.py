from pathlib import Path
import pandas as pd

def load_features(output_dir: str, features: set[str]) -> pd.DataFrame:
    paths = sorted(Path(output_dir).glob('inline-*_atoms.parquet'))
    tables = [pd.read_parquet(path).set_index('atom_index') for path in paths]
    columns = {
        name: table[name] for table in tables
        for name in table.columns.intersection(features) 
    }

    return tables[0][['id', 'x', 'y', 'z']].assign(**columns).sort_index().reset_index()