from multisom import apply_multisom, run_multisom
from pathlib import Path

import numpy as np
import pandas as pd

import hashlib
import json
import os
import sys
import time

WAIT_SECONDS = 1800

def label_atoms(frame, config):
    if config.training != 'reference':
        return _label_per_frame(frame, config)

    # The first frame of the analyzed sequence (lowest selected timestep) is
    # always the trainer, so the reference model is deterministic. VOLT passes
    # the index via --frameIndex; single-frame runs are always index 0.
    store = _ModelStore(config)
    if config.frame_index == 0:
        return _train_reference_model(store, frame, config)

    model = store.wait()
    if 'core' not in model:
        return _label_per_frame(frame, config)

    print(f'labelling with the shared model {store.path.name}', file=sys.stderr)
    raw, layer_columns = apply_multisom(frame, model['core'])
    cluster_id, _ = _dense_ids(raw, known=dict(model['dense_map']))

    return cluster_id, layer_columns

def _label_per_frame(frame, config):
    print(f'training MultiSOM: {len(config.layers)} layer(s), mapping={config.mapping}', file=sys.stderr)
    raw, layer_columns = run_multisom(frame, config.layers, mapping=config.mapping, seed=config.seed)
    cluster_id, _ = _dense_ids(raw)

    return cluster_id, layer_columns

def _train_reference_model(store, frame, config):
    print(f'training the reference model: {store.path.name}', file=sys.stderr)
    model = {}

    try:
        raw, layer_columns, core = run_multisom(frame, config.layers, mapping=config.mapping, seed=config.seed, capture_model=True)
        cluster_id, dense_map = _dense_ids(raw)

        if core is not None:
            model = {
                'core': core,
                'dense_map': list(dense_map.items())
            }

    finally:
        store.publish(model)

    return cluster_id, layer_columns

def _dense_ids(raw, known: dict | None = None):
    '''
    Maps raw SOM labels to contiguous ints numbered by first appearance.
    '''
    values = np.asarray(raw)
    if np.issubdtype(values.dtype, np.floating):
        values = values.round(6)

    labels = pd.Series(values)
    mapping = dict(known) if known else {}

    for value in labels.unique():
        mapping.setdefault(value.item(), len(mapping))

    return labels.map(mapping).to_numpy(np.int32), mapping

class _ModelStore:
    '''
    Model file shared by every frame of one analysis run.
    '''

    def __init__(self, config):
        tag = hashlib.sha256(json.dumps({
            'layers': config.layers,
            'mapping': config.mapping,
            'seed': config.seed,
            'features': sorted(config.features)
        }, sort_keys=True, default=str).encode()).hexdigest()[:16]

        # Frame output dirs (<analysisId>-<index>-<random>) of EVERY analysis are
        # siblings in one daemon-wide dir, so the model is scoped by the analysis
        # prefix: equal configs from other analyses never share a model.
        scope = Path(config.output_dir).name.split('-')[0]
        self.path = Path(config.output_dir).parent / f'.multisom-model-{scope}-{tag}.json'

    def publish(self, model: dict) -> None:
        tmp = self.path.with_suffix(f'.json.tmp-{os.getpid()}')
        tmp.write_text(json.dumps(model))
        tmp.replace(self.path)

    def wait(self) -> dict:
        print(f'waiting up to {WAIT_SECONDS}s for the shared model {self.path.name}', file=sys.stderr)
        deadline = time.monotonic() + WAIT_SECONDS
        while time.monotonic() < deadline:
            if self.path.exists():
                return json.loads(self.path.read_text())
            time.sleep(1)
        return {}
