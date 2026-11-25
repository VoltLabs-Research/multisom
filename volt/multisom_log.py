from functools import singledispatchmethod

from multisom.pipeline import TrainedLayer, UnseenNeurons
from multisom.run_log import RunLog, Stopwatch, format_duration, info, warn
from .labeling import WAIT_SECONDS, AwaitedModel, ModelStore, PublishedModel

import numpy as np
import pandas as pd

MAX_LISTED_CLUSTERS = 20
WAIT_NOTE_SECONDS = 60

class MultisomLog(RunLog):
    '''
    The execution log of one MultiSOM run. Each phase hands it what it built, and
    the log writes what that holds and how long the phase took.
    '''

    def __init__(self, config):
        super().__init__('MultiSOM', [
            ('layers', len(config.layers)),
            ('training', config.training),
            ('mapping', config.mapping),
            ('seed', config.seed),
            ('frameIndex', config.frame_index)
        ])
        self._config = config
        # Labeling spans the wait for the shared model and every layer's training, each a phase of its own.
        self._labeling = Stopwatch()
        self._next_wait_note = WAIT_NOTE_SECONDS
        self._partial_layer = None

    @singledispatchmethod
    def dump(self, result):
        super().dump(result)

    @dump.register
    def _(self, frame: pd.DataFrame):
        config = self._config
        info(f'Loaded {len(frame)} atoms x {len(config.features)} features ({", ".join(sorted(config.features))}) '
             f'from {len(config.feature_paths)} table(s) in {self.phase_elapsed()}')
        self._labeling.restart()

    @dump.register
    def _(self, store: ModelStore):
        # A poll of the store that found no model yet.
        waited = self._labeling.seconds()
        if waited >= self._next_wait_note:
            info(f'  Still waiting for the shared model ({format_duration(waited)})')
            self._next_wait_note += WAIT_NOTE_SECONDS

    @dump.register
    def _(self, model: AwaitedModel):
        waited = self.phase_elapsed()
        if model.content is None:
            warn(f'timed out after {format_duration(WAIT_SECONDS)} waiting for the shared model {model.path.name}')
        else:
            info(f'Shared model found after {waited}')
        if not model.reusable:
            info('No reusable shared model: this frame trains its own MultiSOM')

    @dump.register
    def _(self, layer: TrainedLayer):
        if layer.number == 1:
            if layer.random_weight_soms:
                warn('layer 1 has fewer training atoms than its batch size (%d): its SOM '
                     'keeps its random initial weights' % layer.settings['batch_size'])
            info('Layer 1 trained in %s: %d clusters' % (self.phase_elapsed(), len(layer.found_clusters[0])))
            return

        info('Layer %d trained in %s: %d sub-clusters from %d parent clusters (%d training atoms)'
             % (layer.number, self.phase_elapsed(), sum(len(groups) for groups in layer.found_clusters),
                len(layer.found_clusters), sum(layer.training_rows)))
        skipped = layer.parents - len(layer.found_clusters)
        if skipped:
            info('  %d layer-%d cluster(s) matched no atoms and trained no SOM' % (skipped, layer.number - 1))
            # The first layer that skipped parent clusters is what keeps the reference model from being captured.
            if self._partial_layer is None:
                self._partial_layer = layer
        random_weight_soms = layer.random_weight_soms
        if random_weight_soms:
            warn('%d layer-%d SOM(s) had fewer training atoms than the batch size (%d) and '
                 'kept their random initial weights' % (random_weight_soms, layer.number, layer.settings['batch_size']))

    @dump.register
    def _(self, unseen: UnseenNeurons):
        if unseen.replacements:
            atoms = int(unseen.groups.isin(list(unseen.replacements)).sum())
            warn('%d atom(s) hit SOM neuron(s) unused in the reference frame; remapped them '
                 'to the nearest used neuron: %s' % (atoms, unseen.replacements))

    @dump.register
    def _(self, model: PublishedModel):
        if model.content:
            info(f'Published the reference model {model.path.name}')
        elif self._partial_layer is not None:
            layer = self._partial_layer
            warn('cannot capture a reusable model: layer %d trained %d child '
                 'SOMs for %d observed parent clusters; every frame will train '
                 'on its own' % (layer.number, len(layer.found_clusters), layer.parents))

    @dump.register
    def _(self, cluster_id: np.ndarray):
        counts = np.bincount(cluster_id)
        present = np.flatnonzero(counts)
        present = present[np.argsort(-counts[present], kind='stable')]
        total = len(cluster_id)
        info(f'Labeled {total} atoms into {len(present)} clusters in {self._labeling.elapsed()}')
        for value in present[:MAX_LISTED_CLUSTERS]:
            info(f'  Cluster_{value}: {counts[value]} atoms ({100.0 * counts[value] / total:.1f}%)')
        rest = present[MAX_LISTED_CLUSTERS:]
        if len(rest):
            info(f'  ... and {len(rest)} smaller clusters with {counts[rest].sum()} atoms')
