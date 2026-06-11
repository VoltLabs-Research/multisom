import numpy as np
import pandas as pd

import json

IDENTITY_DTYPES = {
    'atom_index': 'uint32',
    'id': 'uint64',
    'x': 'float64',
    'y': 'float64',
    'z': 'float64'
}

def write_atoms_parquet(output_base: str, frame, cluster_id, layer_columns) -> str:
    label = [f'Cluster_{c}' for c in cluster_id]
    atoms = frame[list(IDENTITY_DTYPES)].astype(IDENTITY_DTYPES).assign(
        bucket=label,
        structure_id=cluster_id,
        structure_name=label,
        cluster_id=cluster_id,
        **{
            name: column.astype('float64') for name, column in layer_columns.items()
        }
    )

    path = f'{output_base}_atoms.parquet'
    atoms.to_parquet(path, index=False, compression='zstd')
    
    return path

def write_summary_parquet(output_base: str, cluster_id, num_layers: int, features: list[str]) -> str:
    unique, counts = np.unique(cluster_id, return_counts=True)
    payload = {
        'main_listing': {
            'total_atoms': len(cluster_id),
            'num_clusters': len(unique),
            'num_layers': num_layers,
            'features': features
        },
        'sub_listings': {
            'clusters': [{
                'cluster_id': int(value),
                'count': int(count)
            } for value, count in zip(unique, counts)]
        }
    }

    path = f'{output_base}_multisom.parquet'
    pd.DataFrame({
        'payload': [json.dumps(payload)]
    }).to_parquet(path, index=False, compression='zstd')

    return path