from multisom import apply_multisom, run_multisom
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

import hashlib
import json
import os
import time

WAIT_SECONDS = 1800

def label_atoms(frame, config, log):
    if config.training != 'reference':
        return _label_per_frame(frame, config, log)

    # The first frame of the analyzed sequence (lowest selected timestep) is
    # always the trainer, so the reference model is deterministic. VOLT passes
    # the index via --frameIndex; single-frame runs are always index 0.
    store = ModelStore(config)
    if config.frame_index == 0:
        return _train_reference_model(store, frame, config, log)

    model = store.wait(log)
    log.dump(model)
    if not model.reusable:
        return _label_per_frame(frame, config, log)

    raw, layer_columns = apply_multisom(frame, model.content['core'], log)
    cluster_id, _ = _dense_ids(raw, known=dict(model.content['dense_map']))

    return cluster_id, layer_columns

def _label_per_frame(frame, config, log):
    raw, layer_columns = run_multisom(frame, config.layers, log, mapping=config.mapping, seed=config.seed)
    cluster_id, _ = _dense_ids(raw)

    return cluster_id, layer_columns

def _train_reference_model(store, frame, config, log):
    model = {}

    try:
        raw, layer_columns, core = run_multisom(frame, config.layers, log, mapping=config.mapping, seed=config.seed, capture_model=True)
        cluster_id, dense_map = _dense_ids(raw)

        if core is not None:
            model = {
                'core': core,
                'dense_map': list(dense_map.items())
            }

    finally:
        store.publish(model)

    log.dump(PublishedModel(store.path, model))
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

class PublishedModel(NamedTuple):
    '''
    The reference model the first frame left for the others: empty when none could be captured.
    '''
    path: Path
    content: dict

class AwaitedModel(NamedTuple):
    '''
    The reference model a later frame waited for: None when it never appeared.
    '''
    path: Path
    content: dict | None

    @property
    def reusable(self) -> bool:
        return self.content is not None and 'core' in self.content

class ModelStore:
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

        # VOLT's output base of a frame is its output dir (<analysisId>-<index>-<random>),
        # and those of EVERY analysis are siblings in one daemon-wide dir, so the model is
        # scoped by the analysis prefix: equal configs from other analyses never share a model.
        base = Path(config.output_base)
        scope = base.name.split('-')[0]
        self.path = base.parent / f'.multisom-model-{scope}-{tag}.json'

    def publish(self, model: dict) -> None:
        tmp = self.path.with_suffix(f'.json.tmp-{os.getpid()}')
        tmp.write_text(json.dumps(model))
        tmp.replace(self.path)

    def wait(self, log) -> AwaitedModel:
        # Each poll that finds no model goes to the log, which notes the wait now and then.
        deadline = time.monotonic() + WAIT_SECONDS
        while time.monotonic() < deadline:
            if self.path.exists():
                return AwaitedModel(self.path, json.loads(self.path.read_text()))
            time.sleep(1)
            log.dump(self)
        return AwaitedModel(self.path, None)
