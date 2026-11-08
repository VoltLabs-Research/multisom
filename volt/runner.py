from dataclasses import dataclass, field

from multisom.log import Stopwatch, log
from multisom.progress import ProgressSteps, job_progress
from .features import load_features
from .interface import OPTIONS
from . import labeling, outputs

import numpy as np
import argparse
import json

MAX_LISTED_CLUSTERS = 20

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
    run_timer = Stopwatch()
    job_progress.start()
    log(f'Parameters: layers={len(config.layers)}, training={config.training}, mapping={config.mapping}, '
        f'seed={config.seed}, frameIndex={config.frame_index}')

    # Reading the features, training and labeling, then the outputs.
    with ProgressSteps(1, 8, 1) as steps:
        job_progress.set_text('Reading the feature tables')
        timer = Stopwatch()
        frame = load_features(config.feature_paths, config.features)
        log(f'Loaded {len(frame)} atoms x {len(config.features)} features ({", ".join(sorted(config.features))}) '
            f'from {len(config.feature_paths)} table(s) in {timer.elapsed()}')
        steps.next('Labeling atoms with the MultiSOM')

        timer.restart()
        cluster_id, layer_columns = labeling.label_atoms(frame, config)
        _log_clusters(cluster_id, timer)
        steps.next('Writing outputs')
        atoms_path = outputs.write_atoms_parquet(config.output_base, frame, cluster_id, layer_columns)
        log(f'Exported atoms data to: {atoms_path}')
        summary_path = outputs.write_summary_parquet(config.output_base, cluster_id, len(layer_columns), sorted(config.features))
        log(f'Summary parquet written to {summary_path}')

    result = {
        'ok': True,
        'outputBase': config.output_base,
        'outputs': [atoms_path, summary_path],
        'num_clusters': len(np.unique(cluster_id))
    }
    log(f'MultiSOM completed in {run_timer.elapsed()}.')
    job_progress.finish()
    return result

def _log_clusters(cluster_id, timer):
    counts = np.bincount(cluster_id)
    present = np.flatnonzero(counts)
    present = present[np.argsort(-counts[present], kind='stable')]
    total = len(cluster_id)
    log(f'Labeled {total} atoms into {len(present)} clusters in {timer.elapsed()}')
    for value in present[:MAX_LISTED_CLUSTERS]:
        log(f'  Cluster_{value}: {counts[value]} atoms ({100.0 * counts[value] / total:.1f}%)')
    rest = present[MAX_LISTED_CLUSTERS:]
    if len(rest):
        log(f'  ... and {len(rest)} smaller clusters with {counts[rest].sum()} atoms')
