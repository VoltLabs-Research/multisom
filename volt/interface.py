from __future__ import annotations

OPTIONS = (
    {
        'flag': '--features',
        'type': 'path-list',
        'help': 'Per-atom parquet tables supplying the feature columns.',
    },
    {
        'flag': '--layers',
        'type': 'string',
        'help': 'JSON array of SOM layer definitions.',
    },
    {
        'flag': '--training',
        'type': 'enum',
        'help': 'Whether the map trains on a reference frame or per frame.',
        'default': 'per_frame',
        'values': ['reference', 'per_frame'],
    },
    {
        'flag': '--mapping',
        'type': 'enum',
        'help': 'Cluster-to-label mapping.',
        'default': 'linear',
        'values': ['linear', 'log'],
    },
    {'flag': '--seed', 'type': 'int', 'help': 'Random seed.', 'default': '1982'},
    {'flag': '--frameIndex', 'type': 'int', 'help': 'Index of the frame being analysed.', 'default': '0'},
)

DESCRIPTOR = {
    'descriptor': 1,
    'name': 'multisom',
    'description': 'Multi-layer self-organising map clustering',
    'frameMode': 'single',
    'needsReferenceFrame': False,
    'positional': ['input', 'output_base'],
    'options': list(OPTIONS),
}
