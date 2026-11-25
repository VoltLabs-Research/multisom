from dataclasses import dataclass, field

from multisom.progress import ProgressSteps, job_progress
from .features import load_features
from .interface import OPTIONS
from .multisom_log import MultisomLog
from . import labeling, outputs

import numpy as np
import argparse
import json

@dataclass(frozen=True)
class RunConfig:
    input_path: str
    output_base: str
    layers: list[dict]
    training: str
    mapping: str
    seed: int
    frame_index: int
    feature_paths: list[str] = field(default_factory=list)

    @property
    def features(self) -> set[str]:
        return {
            feature for layer in self.layers for feature in layer['features']
        }

def _split_paths(raw: str) -> list[str]:
    return [part for part in raw.replace(',', ' ').split() if part]

_PARSERS = {
    '--features': dict(type=_split_paths, default=[], dest='feature_paths'),
    '--layers': dict(type=json.loads, required=True),
    '--training': dict(default='per_frame'),
    '--mapping': dict(default='linear'),
    '--seed': dict(type=int, default=1982),
    '--frameIndex': dict(type=int, default=0, dest='frame_index'),
}

def parse_args(args: list[str]) -> RunConfig:
    parser = argparse.ArgumentParser(prog='multisom', add_help=False)
    parser.add_argument('input_path')
    parser.add_argument('output_base')
    for option in OPTIONS:
        parser.add_argument(option['flag'], **_PARSERS[option['flag']])

    namespace, _ = parser.parse_known_args(args)

    return RunConfig(**vars(namespace))

def run(args: list[str]) -> dict:
    config = parse_args(args)
    job_progress.start()
    log = MultisomLog(config)

    # Reading the features, training and labeling, then the outputs.
    with ProgressSteps(
        ('Reading the feature tables', 1),
        ('Labeling atoms with the MultiSOM', 8),
        ('Writing outputs', 1),
    ) as steps:
        frame = load_features(config.feature_paths, config.features)
        log.dump(frame)
        steps.next()

        cluster_id, layer_columns = labeling.label_atoms(frame, config, log)
        log.dump(cluster_id)
        steps.next()
        atoms_path = outputs.write_atoms_parquet(config.output_base, frame, cluster_id, layer_columns)
        summary_path = outputs.write_summary_parquet(config.output_base, cluster_id, len(layer_columns), sorted(config.features))

    result = {
        'ok': True,
        'outputBase': config.output_base,
        'outputs': [atoms_path, summary_path],
        'num_clusters': len(np.unique(cluster_id))
    }
    log.dump(result)
    job_progress.finish()
    return result
