from dataclasses import dataclass

from .features import load_features
from . import labeling, outputs

import numpy as np
import argparse
import json


@dataclass(frozen=True)
class RunConfig:
    output_dir: str
    layers: list[dict]
    training: str
    mapping: str
    seed: int
    frame_index: int

    @property
    def features(self) -> set[str]:
        return {
            feature for layer in self.layers for feature in layer['features']
        }
    
def parse_args(args: list[str]) -> RunConfig:
    parser = argparse.ArgumentParser(prog='multisom', add_help=False)
    parser.add_argument('output_dir')
    parser.add_argument('--layers', type=json.loads, required=True)
    parser.add_argument('--training', default='per_frame')
    parser.add_argument('--mapping', default='linear')
    parser.add_argument('--seed', type=int, default=1982)
    parser.add_argument('--frameIndex', type=int, default=0, dest='frame_index')

    namespace, _ = parser.parse_known_args(args)

    return RunConfig(**vars(namespace))

def run(args: list[str]) -> dict:
    config = parse_args(args)
    frame = load_features(config.output_dir, config.features)
    cluster_id, layer_columns = labeling.label_atoms(frame, config)

    return {
        'ok': True,
        'outputBase': config.output_dir,
        'outputs': [
            outputs.write_atoms_parquet(config.output_dir, frame, cluster_id, layer_columns),
            outputs.write_summary_parquet(config.output_dir, cluster_id, len(layer_columns), sorted(config.features))
        ],
        'num_clusters': len(np.unique(cluster_id))
    }