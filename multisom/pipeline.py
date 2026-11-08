"""
MultiSOM core pipeline — refactored into a side-effect-free function for the
VOLT plugin entrypoint.

This module extracts the training/labelling logic from the original ``main.py``
(Copyright 2021 Franco Aquistapace, Apache-2.0) and exposes it as a pure
function ``run_multisom(df, layers, ...)`` that:

  * takes a pandas DataFrame whose columns already contain the requested
    feature columns (e.g. ``csp``, ``coordination``),
  * trains the hierarchical/divisive multilayer Self-Organizing Map exactly as
    the original code did, and
  * returns the per-atom cluster assignment for every layer.

It performs NO file I/O, sets NO global RNG seed implicitly (seed is a
parameter), and does not depend on ``input_params``/OVITO. Logs go to STDERR so
that any STDOUT channel the host reserves stays clean.

The SOM math itself lives untouched in ``multisom.som``.
"""

import copy

import numpy as np
import pandas as pd

from .log import Stopwatch, log, warn
from .progress import job_progress
from .som import SOM, SOM_Batched


# ----------------------------------------------------------------------------
# Feature scaling (verbatim behaviour from main.py:normalize_data)
# ----------------------------------------------------------------------------
def normalize_data(df, features, scaling, min_max_features, training):
    """Scale ``df[features]`` according to ``scaling`` (per-feature).

    ``scaling`` is a list parallel to ``features`` with values
    ``'normal'`` (min-max), ``'standard'`` (z-score) or ``'robust'``
    (median/IQR). When ``training`` is True the per-feature statistics are
    computed from ``df`` and appended to ``min_max_features``; otherwise they
    are read back from it (index-aligned with ``features``).
    """
    norm_df = df[features].copy()
    if training:
        for i in range(len(features)):
            feat = features[i]
            scale = scaling[i]
            if scale == 'normal':
                min_value = norm_df[feat].min()
                max_value = norm_df[feat].max()
                delta = max_value - min_value
                if delta == 0:
                    delta = 1
                norm_df[feat] = (norm_df[feat] - min_value) / delta
                min_max_features.append([min_value, max_value])

            elif scale == 'standard':
                mean = norm_df[feat].mean()
                std = norm_df[feat].std()
                if std == 0:
                    std = 1
                norm_df[feat] = (norm_df[feat] - mean) / std
                min_max_features.append([mean, std])

            elif scale == 'robust':
                q25 = df[feat].describe()['25%']
                q75 = df[feat].describe()['75%']
                iqr = q75 - q25
                if iqr == 0:
                    iqr = 1
                median = df[feat].median()
                norm_df[feat] = (norm_df[feat] - median) / iqr
                min_max_features.append([iqr, median])

            else:
                raise ValueError(
                    "scaling values must be either 'standard', 'robust' or 'normal'"
                )
    else:
        for i in range(len(features)):
            feat = features[i]
            scale = scaling[i]
            if scale == 'normal':
                min_value = min_max_features[i][0]
                max_value = min_max_features[i][1]
                delta = max_value - min_value
                if delta == 0:
                    delta = 1
                norm_df[feat] = (norm_df[feat] - min_value) / delta

            elif scale == 'standard':
                mean = min_max_features[i][0]
                std = min_max_features[i][1]
                if std == 0:
                    std = 1
                norm_df[feat] = (norm_df[feat] - mean) / std

            elif scale == 'robust':
                iqr = min_max_features[i][0]
                median = min_max_features[i][1]
                if iqr == 0:
                    iqr = 1
                norm_df[feat] = (norm_df[feat] - median) / iqr

    return norm_df


# ----------------------------------------------------------------------------
# Cluster-id mapping schemes (verbatim from main.py)
# ----------------------------------------------------------------------------
def generate_prime(n):
    primes = []
    m = 2
    while len(primes) < n:
        add = True
        for i in primes:
            if m % i == 0:
                add = False
        if add:
            primes.append(m)
        m += 1
    return primes[-1]


def map_to_cluster_linear(groups_df, c_id, c_next_id,
                          use_int=False, max_int=None, min_int=None, N=1):
    """Interpolate child ids into the numeric gap between consecutive parents."""
    if not use_int:
        max_val, min_val = groups_df.max(), groups_df.min()
        if max_val - min_val == 0:
            delta = c_next_id - c_id
            lim = (N - 1) / N
            mapped_df = c_id + delta * lim * groups_df.copy() / (N - 1)
        else:
            delta = c_next_id - c_id
            lim = (max_val / (max_val + 1))
            mapped_df = c_id + \
                delta * lim * (groups_df.copy() - min_val) / (max_val - min_val)
        return mapped_df
    else:
        max_val, min_val = max_int, min_int
        delta = c_next_id - c_id
        lim = (max_val / (max_val + 1))
        mapped_df = c_id + delta * lim * (groups_df - min_val) / (max_val - min_val)
        return mapped_df


def map_to_cluster_godel(prev_id, layer, group):
    """Prime-power Gödel numbering: ``new_id = prev_id * p_layer ** group``."""
    p_i = generate_prime(layer)
    new_id = prev_id * (p_i ** group)
    return new_id


def map_to_cluster_default(groups_df):
    """Identity mapping (plain per-layer integers, no ancestry encoding)."""
    return groups_df


# ----------------------------------------------------------------------------
# Layer-config normalisation
# ----------------------------------------------------------------------------
_VALID_SCALES = ('normal', 'standard', 'robust')
_VALID_MAPPINGS = ('linear', 'godel', 'default')


def normalize_layers(layers):
    """Accept both the original capitalised keys and the plugin's lowercase keys.

    Returns a list of canonical layer dicts with keys:
    ``features`` (list[str]), ``scaling`` (list[str] parallel to features),
    ``f`` (float), ``sigma`` (float), ``eta`` (float), ``N`` (int),
    ``batched`` (bool), ``batch_size`` (int).

    A scalar ``scaling`` is broadcast across all features.
    """
    canonical = []
    for raw in layers:
        features = list(raw['features'])
        if not features:
            raise ValueError('At least one feature must be specified for each layer')

        scaling = raw.get('scaling', 'normal')
        if isinstance(scaling, str):
            scaling = [scaling] * len(features)
        else:
            scaling = list(scaling)
        if len(scaling) != len(features):
            raise ValueError('scaling and features must be the same length for each layer')
        for s in scaling:
            if s not in _VALID_SCALES:
                raise ValueError("scaling values must be one of %s" % (_VALID_SCALES,))

        f = raw.get('f', 1)
        f = int(1) if str(f) == '1' else float(f)
        if f > 1 or f <= 0:
            raise ValueError('f must be such that 0 < f <= 1 for every layer')

        N = raw.get('N', 5)
        if int(N) != N:
            raise ValueError('N must be an int for every layer')
        N = int(N)
        if N < 2:
            raise ValueError('N (number of output neurons) must be >= 2 for every layer')

        batched = raw.get('batched', raw.get('Batched', True))
        if not isinstance(batched, bool):
            raise ValueError('batched must be either True or False for every layer')

        canonical.append({
            'features': features,
            'scaling': scaling,
            'f': f,
            'sigma': float(raw.get('sigma', 0.5)),
            'eta': float(raw.get('eta', 0.5)),
            'N': N,
            'batched': batched,
            'batch_size': int(raw.get('batch_size', raw.get('Batch_size', 100))),
        })
    return canonical


def _describe_layer(layer):
    """Inputs and hyperparameters of a canonical layer, for its log line."""
    features, scaling = layer['features'], layer['scaling']
    if len(set(scaling)) == 1:
        inputs = '%s (%s scaling)' % (', '.join(features), scaling[0])
    else:
        inputs = ', '.join('%s (%s)' % pair for pair in zip(features, scaling))
    mode = 'batch size %d' % layer['batch_size'] if layer['batched'] else 'online'
    return '%s; N=%d, sigma=%s, eta=%s, f=%s, %s' % (
        inputs, layer['N'], layer['sigma'], layer['eta'], layer['f'], mode)


def _keeps_random_weights(som, rows):
    """SOM_Batched.train runs ``rows // batch_size`` batches: none below one batch."""
    return isinstance(som, SOM_Batched) and rows < som.batch_size


# ----------------------------------------------------------------------------
# Model (de)serialization helpers — used by the reference-frame training mode
# ----------------------------------------------------------------------------
def _to_jsonable(value):
    """Recursively convert numpy scalars/arrays so the model can be JSON-dumped."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _som_from_weights(W1):
    """Rebuild a predict-only SOM from a stored weight matrix.

    Only ``W1`` matters for :meth:`SOM.predict`; sigma/eta are training-time
    hyperparameters, so placeholders are fine here.
    """
    som = SOM(sigma=1.0, eta=1.0, size=[W1.shape[0], W1.shape[1]])
    som.W1 = W1
    return som


def _remap_unseen_groups(groups, observed, W1):
    """Snap SOM groups that the reference frame never produced onto seen ones.

    A new frame can activate an output neuron that had no atoms in the
    reference frame; the stored model has no mapping (and no child SOM) for it.
    Reassign those atoms to the *nearest seen neuron* in weight space, which is
    the most similar cluster by construction.
    """
    observed_ints = [int(g) for g in observed]
    values = groups.astype(int)
    unseen = sorted(set(int(v) for v in values.unique()) - set(observed_ints))
    if not unseen:
        return values

    W = np.asarray(W1, dtype=float)
    observed_weights = W[:, observed_ints]
    replacements = {}
    for g in unseen:
        distances = np.linalg.norm(observed_weights - W[:, [g]], axis=0)
        replacements[g] = observed_ints[int(np.argmin(distances))]
    warn('%d atom(s) hit SOM neuron(s) unused in the reference frame; remapped them '
         'to the nearest used neuron: %s' % (int(values.isin(unseen).sum()), replacements))
    return values.replace(replacements)


def _capture_model(mapping, LAYERS, soms, all_min_max_features,
                   found_clusters, mapped_found_clusters, first_layer_mapping):
    """Snapshot everything :func:`apply_multisom` needs to relabel new frames.

    Returns None when no replayable snapshot exists: apply_multisom walks
    children with a flat (j, k) index that assumes every observed parent
    cluster trained a child SOM, but the training loop skips parents that
    matched zero rows (godel's positional parent mapping + an unused layer-1
    neuron). The labels of the frame that trained are still valid.
    """
    for i in range(1, len(LAYERS)):
        expected_children = sum(len(group) for group in found_clusters[i - 1])
        if len(found_clusters[i]) != expected_children or len(soms[i]) != expected_children:
            warn('cannot capture a reusable model: layer %d trained %d child '
                 'SOMs for %d observed parent clusters; every frame will train '
                 'on its own' % (i + 1, len(found_clusters[i]), expected_children))
            return None

    return {
        'version': 1,
        'mapping': mapping,
        'layers': _to_jsonable(LAYERS),
        'som_weights': [[som.W1.tolist() for som in layer_soms] for layer_soms in soms],
        'min_max': _to_jsonable(all_min_max_features),
        'found_clusters': _to_jsonable(found_clusters),
        'mapped_found_clusters': _to_jsonable(mapped_found_clusters),
        'first_layer_mapping': [
            [_to_jsonable(group), _to_jsonable(mapped)]
            for group, mapped in first_layer_mapping.items()
        ]
    }


# ----------------------------------------------------------------------------
# The public entrypoint
# ----------------------------------------------------------------------------
def run_multisom(df, layers, mapping='linear', seed=1982, capture_model=False):
    """Train the multilayer SOM on ``df`` and label every atom per layer.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-atom table that already contains every feature column referenced by
        ``layers`` (e.g. ``csp``, ``coordination``). Row order is preserved.
    layers : list of dict
        SOM layer stack (see :func:`normalize_layers` for accepted keys).
    mapping : str
        Cluster-id mapping scheme: ``'linear'`` (default), ``'godel'`` or
        ``'default'``.
    seed : int or None
        RNG seed for reproducible weight initialisation. ``None`` leaves the
        global NumPy RNG untouched.
    capture_model : bool
        When True, also return a JSON-serializable snapshot of the trained
        model (weights, scaling stats and cluster-id mappings) that
        :func:`apply_multisom` can use to relabel other frames consistently —
        or None when no replayable snapshot exists (no exception is raised).

    Returns
    -------
    (cluster_id, layer_columns) or (cluster_id, layer_columns, model)
        ``cluster_id`` is a numpy array (one entry per row of ``df``) holding the
        deepest-layer cluster assignment — the per-atom ``cluster_id``.
        ``layer_columns`` is a ``dict`` ``{'layer_1': Series, ..., 'layer_n': Series}``
        index-aligned with ``df`` for callers that want the full hierarchy.
        ``model`` (only when ``capture_model``) is the reusable snapshot, or
        None when one can't be captured.
    """
    if mapping not in _VALID_MAPPINGS:
        raise ValueError("mapping must be one of %s" % (_VALID_MAPPINGS,))
    if seed is not None:
        np.random.seed(seed)

    LAYERS = normalize_layers(layers)
    if not LAYERS:
        raise ValueError('At least one layer must be specified')

    # Validate that every requested feature exists in df
    requested = sorted({feat for layer in LAYERS for feat in layer['features']})
    missing = [feat for feat in requested if feat not in df.columns]
    if missing:
        raise ValueError(
            'Requested features not present in the data: %s. Available: %s'
            % (missing, sorted(df.columns))
        )

    MAPPING = mapping
    ALL_FEATURES = [layer['features'] for layer in LAYERS]
    ALL_SCALES = [layer['scaling'] for layer in LAYERS]
    SIGMAS = [layer['sigma'] for layer in LAYERS]
    ETAS = [layer['eta'] for layer in LAYERS]
    BATCHED = [layer['batched'] for layer in LAYERS]
    BATCH_SIZES = [layer['batch_size'] for layer in LAYERS]
    fs = [layer['f'] for layer in LAYERS]

    # Output sizes; single-feature layers need a 2-dim input (aux_col duplicate)
    SIZES = []
    for i in range(len(LAYERS)):
        if len(ALL_FEATURES[i]) == 1:
            SIZES.append([2, LAYERS[i]['N']])
        else:
            SIZES.append([len(ALL_FEATURES[i]), LAYERS[i]['N']])

    all_min_max_features = []
    found_clusters = []
    soms = []

    def _make_som(i):
        if BATCHED[i]:
            return SOM_Batched(sigma=SIGMAS[i], eta=ETAS[i],
                               size=SIZES[i], batch_size=BATCH_SIZES[i])
        return SOM(sigma=SIGMAS[i], eta=ETAS[i], size=SIZES[i])

    # --- Layer 1 -------------------------------------------------------------
    log('Training MultiSOM: %d layer(s) on %d atoms' % (len(LAYERS), len(df)))
    # Every layer labels all the atoms once, so each is an equal share of the training.
    job_progress.begin_steps([1.0] * len(LAYERS))
    job_progress.set_text('Training layer 1 of %d' % len(LAYERS))
    soms.append([_make_som(0)])

    og_df = df.reset_index(drop=True)
    min_max_temp = []
    og_training_df = normalize_data(df=og_df, features=ALL_FEATURES[0],
                                    scaling=ALL_SCALES[0],
                                    min_max_features=min_max_temp,
                                    training=True)
    all_min_max_features.append([min_max_temp.copy()])

    # A single-feature layer needs a 2-dim input (SIZES forces inputLayerSize=2),
    # so duplicate the lone feature into an auxiliary column. The original code
    # only did this for secondary layers; with user-chosen features the first
    # layer can now be single-feature too.
    if len(ALL_FEATURES[0]) == 1:
        og_training_df['aux_col'] = og_training_df[ALL_FEATURES[0][0]]

    f = fs[0]
    training_df = og_training_df.sample(frac=f)

    log('Layer 1: 1 SOM on %d training atoms; %s' % (len(training_df), _describe_layer(LAYERS[0])))
    if _keeps_random_weights(soms[0][0], len(training_df)):
        warn('layer 1 has fewer training atoms than its batch size (%d): its SOM '
             'keeps its random initial weights' % BATCH_SIZES[0])
    timer = Stopwatch()
    soms[0][0].train(training_df, progress_label='  Training layer 1 SOM')

    norm_df = normalize_data(df=og_df, features=ALL_FEATURES[0],
                             scaling=ALL_SCALES[0],
                             min_max_features=all_min_max_features[0][0],
                             training=False)
    if len(ALL_FEATURES[0]) == 1:
        norm_df['aux_col'] = norm_df[ALL_FEATURES[0][0]]
    results = soms[0][0].predict(norm_df)
    result_cols = results.columns.to_list()
    groups = results[result_cols[-1]]
    unique_groups = list(groups.unique())
    found_clusters.append([unique_groups])
    mapped_found_clusters = copy.deepcopy(found_clusters)
    log('Layer 1 trained in %s: %d clusters' % (timer.elapsed(), len(unique_groups)))

    first_layer_mapping = {}
    if MAPPING == 'godel':
        for j in range(len(found_clusters[0])):
            for k in range(len(found_clusters[0][j])):
                mapped_found_clusters[0][j][k] = map_to_cluster_godel(
                    prev_id=1, layer=1, group=k + 1)
        mapped_groups = map_to_cluster_godel(prev_id=1, layer=1, group=groups + 1)
    else:  # linear or default
        for k in range(len(unique_groups)):
            first_layer_mapping[unique_groups[k]] = k
            mapped_found_clusters[0][0][k] = k
        mapped_groups = groups.map(first_layer_mapping)

    new_df = pd.concat([og_df, mapped_groups], axis=1)
    col_list = new_df.columns.to_list()
    new_df = new_df.rename(columns={col_list[-1]: 'layer_1'})

    # --- Secondary layers ----------------------------------------------------
    for i in range(len(LAYERS) - 1):
        job_progress.next_step('Training layer %d of %d' % (i + 2, len(LAYERS)))
        parents = sum(len(group) for group in found_clusters[i])
        log('Layer %d: 1 SOM per layer-%d cluster (%d SOMs); %s'
            % (i + 2, i + 1, parents, _describe_layer(LAYERS[i + 1])))
        # One step per SOM of the layer, so each one's training fills its own share.
        job_progress.begin_steps([1.0] * max(1, parents))
        visited_clusters = 0
        timer = Stopwatch()
        trained = empty = random_weights = training_rows = 0
        layer_name = 'layer_' + str(i + 2)
        new_df[layer_name] = 0.0 if MAPPING == 'linear' else 0
        prev_layer_name = 'layer_' + str(i + 1)

        layer_found_clusters = []
        som_layer_next = []
        min_max_temp = []

        for j in range(len(found_clusters[i])):
            k_per_j = len(found_clusters[i][j])
            som_group_next = []
            min_max_features_group_next = [[] for _ in range(len(found_clusters[i][j]))]

            for k in range(len(found_clusters[i][j])):
                som_group_next.append(_make_som(i + 1))

            f_next = fs[i + 1]

            for k in range(len(found_clusters[i][j])):
                if visited_clusters:
                    job_progress.next_step()
                visited_clusters += 1
                cluster_id = mapped_found_clusters[i][j][k]
                cluster_data = new_df.loc[new_df[prev_layer_name] == cluster_id].copy()
                if cluster_data.shape[0] == 0:
                    empty += 1
                    continue

                som_id = int(k_per_j * j + k)
                cluster_data = normalize_data(cluster_data.copy(),
                                              ALL_FEATURES[i + 1],
                                              ALL_SCALES[i + 1],
                                              min_max_features_group_next[k],
                                              training=True)

                if len(ALL_FEATURES[i + 1]) == 1:
                    if isinstance(cluster_data, pd.Series):
                        cluster_data = cluster_data.to_frame()
                    cluster_data['aux_col'] = cluster_data[ALL_FEATURES[i + 1][0]]

                training_data = cluster_data.sample(frac=f_next)
                trained += 1
                training_rows += len(training_data)
                if _keeps_random_weights(som_group_next[k], len(training_data)):
                    random_weights += 1
                som_group_next[k].train(training_data, progress_label='  Training layer %d SOM %d/%d'
                                        % (i + 2, trained + empty, parents))
                cluster_results = som_group_next[k].predict(cluster_data)

                cluster_result_cols = cluster_results.columns.to_list()
                cluster_groups = cluster_results[cluster_result_cols[-1]]
                layer_found_clusters.append(list(cluster_groups.unique()))

                if MAPPING == 'godel':
                    final_cluster_groups = map_to_cluster_godel(
                        prev_id=cluster_id, layer=i + 2, group=cluster_groups + 1)
                elif MAPPING == 'linear':
                    j_cond = (j == len(found_clusters[i]) - 1)
                    k_cond = (k == len(found_clusters[i][j]) - 1)
                    if j_cond and k_cond:
                        prev_cluster_id = mapped_found_clusters[i][j][k - 1]
                        diff = abs(prev_cluster_id - cluster_id)
                        next_cluster_id = cluster_id + diff
                    elif k_cond and not j_cond:
                        next_cluster_id = mapped_found_clusters[i][j + 1][0]
                    else:
                        next_cluster_id = mapped_found_clusters[i][j][k + 1]
                    final_cluster_groups = map_to_cluster_linear(
                        groups_df=cluster_groups,
                        c_id=cluster_id,
                        c_next_id=next_cluster_id)
                else:  # default
                    final_cluster_groups = map_to_cluster_default(cluster_groups)

                new_df.loc[new_df[prev_layer_name] == cluster_id, layer_name] = \
                    final_cluster_groups

            som_layer_next.extend(som_group_next)
            min_max_temp.extend(min_max_features_group_next)

        soms.append(som_layer_next)
        all_min_max_features.append(min_max_temp)
        found_clusters.append(copy.deepcopy(layer_found_clusters))
        job_progress.end_steps()
        log('Layer %d trained in %s: %d sub-clusters from %d parent clusters (%d training atoms)'
            % (i + 2, timer.elapsed(), sum(len(groups) for groups in layer_found_clusters),
               trained, training_rows))
        if empty:
            log('  %d layer-%d cluster(s) matched no atoms and trained no SOM' % (empty, i + 1))
        if random_weights:
            warn('%d layer-%d SOM(s) had fewer training atoms than the batch size (%d) and '
                 'kept their random initial weights' % (random_weights, i + 2, BATCH_SIZES[i + 1]))

        # Map the freshly found clusters for use as parents of the next layer
        layer_mapped_found_clusters = copy.deepcopy(layer_found_clusters)
        if MAPPING == 'godel':
            for j in range(len(found_clusters[i + 1])):
                for k in range(len(found_clusters[i + 1][j])):
                    prev_total = len(found_clusters[i + 1])
                    prev_classes = len(found_clusters[i])
                    tot_per_class = int(prev_total / prev_classes)
                    prev_id = mapped_found_clusters[i][j // tot_per_class][j % tot_per_class]
                    layer_mapped_found_clusters[j][k] = map_to_cluster_godel(
                        prev_id=prev_id, layer=i + 2, group=k + 1)
        elif MAPPING == 'linear':
            for j in range(len(found_clusters[i + 1])):
                for k in range(len(found_clusters[i + 1][j])):
                    prev_total = len(found_clusters[i + 1])
                    prev_classes = len(found_clusters[i])
                    tot_per_class = int(prev_total / prev_classes)
                    cluster_id = mapped_found_clusters[i][j // tot_per_class][j % tot_per_class]

                    j_cond = (j // tot_per_class == prev_classes - 1)
                    k_cond = (j % tot_per_class ==
                              len(mapped_found_clusters[i][j // tot_per_class]) - 1)
                    if j_cond and k_cond:
                        prev_cluster_id = mapped_found_clusters[i][
                            j // tot_per_class][j % tot_per_class - 1]
                        diff = abs(prev_cluster_id - cluster_id)
                        next_cluster_id = cluster_id + diff
                    elif k_cond and not j_cond:
                        next_cluster_id = mapped_found_clusters[i][j // tot_per_class + 1][0]
                    else:
                        next_cluster_id = mapped_found_clusters[i][
                            j // tot_per_class][j % tot_per_class + 1]

                    max_int = max(found_clusters[i + 1][j])
                    min_int = min(found_clusters[i + 1][j])
                    layer_mapped_found_clusters[j][k] = map_to_cluster_linear(
                        groups_df=found_clusters[i + 1][j][k],
                        c_id=cluster_id,
                        c_next_id=next_cluster_id,
                        use_int=True,
                        max_int=max_int,
                        min_int=min_int)
        else:  # default
            for j in range(len(found_clusters[i + 1])):
                for k in range(len(found_clusters[i + 1][j])):
                    layer_mapped_found_clusters[j][k] = map_to_cluster_default(
                        found_clusters[i + 1][j][k])

        mapped_found_clusters.append(copy.deepcopy(layer_mapped_found_clusters))
    job_progress.end_steps()

    # --- Assemble outputs ----------------------------------------------------
    layer_columns = {}
    n_layers = len(LAYERS)
    for idx in range(n_layers):
        name = 'layer_' + str(idx + 1)
        layer_columns[name] = new_df[name]

    deepest = 'layer_' + str(n_layers)
    cluster_id = new_df[deepest].to_numpy()
    if capture_model:
        model = _capture_model(MAPPING, LAYERS, soms, all_min_max_features,
                               found_clusters, mapped_found_clusters,
                               first_layer_mapping)
        return cluster_id, layer_columns, model
    return cluster_id, layer_columns


# ----------------------------------------------------------------------------
# Relabel a frame with a previously trained model (reference-frame mode)
# ----------------------------------------------------------------------------
def _child_group_lookup(mapping, i, j, k, found_clusters,
                        mapped_found_clusters, observed_groups, cluster_id):
    """Map a child SOM's group indices to the reference frame's cluster ids.

    Re-evaluates the SAME mapping functions the training loop used, on the
    reference frame's observed groups, so a given neuron index always maps to
    the exact id the reference produced (bit-for-bit, which the float-equality
    parent selection relies on).
    """
    groups = pd.Series(observed_groups)
    if mapping == 'godel':
        written = map_to_cluster_godel(prev_id=cluster_id, layer=i + 2,
                                       group=groups + 1)
    elif mapping == 'linear':
        j_cond = (j == len(found_clusters[i]) - 1)
        k_cond = (k == len(found_clusters[i][j]) - 1)
        if j_cond and k_cond:
            prev_cluster_id = mapped_found_clusters[i][j][k - 1]
            diff = abs(prev_cluster_id - cluster_id)
            next_cluster_id = cluster_id + diff
        elif k_cond and not j_cond:
            next_cluster_id = mapped_found_clusters[i][j + 1][0]
        else:
            next_cluster_id = mapped_found_clusters[i][j][k + 1]
        written = map_to_cluster_linear(groups_df=groups, c_id=cluster_id,
                                        c_next_id=next_cluster_id)
    else:  # default
        written = groups
    return dict(zip(list(observed_groups), list(written)))


def apply_multisom(df, model):
    """Label every atom of ``df`` with a model trained by :func:`run_multisom`.

    Runs only the predict/mapping path — no training, no RNG — so every frame
    labelled with the same model gets a consistent ``cluster_id`` space.

    Returns ``(cluster_id, layer_columns)`` exactly like :func:`run_multisom`.
    """
    mapping = model['mapping']
    LAYERS = model['layers']
    som_weights = model['som_weights']
    min_max = model['min_max']
    found_clusters = model['found_clusters']
    mapped_found_clusters = model['mapped_found_clusters']
    first_layer_mapping = {group: mapped for group, mapped in model['first_layer_mapping']}

    requested = sorted({feat for layer in LAYERS for feat in layer['features']})
    missing = [feat for feat in requested if feat not in df.columns]
    if missing:
        raise ValueError(
            'Requested features not present in the data: %s. Available: %s'
            % (missing, sorted(df.columns))
        )

    ALL_FEATURES = [layer['features'] for layer in LAYERS]
    ALL_SCALES = [layer['scaling'] for layer in LAYERS]
    og_df = df.reset_index(drop=True)

    # --- Layer 1 (predict-only mirror of the training path) -------------------
    norm_df = normalize_data(df=og_df, features=ALL_FEATURES[0],
                             scaling=ALL_SCALES[0],
                             min_max_features=min_max[0][0],
                             training=False)
    if len(ALL_FEATURES[0]) == 1:
        norm_df['aux_col'] = norm_df[ALL_FEATURES[0][0]]

    first_W1 = np.asarray(som_weights[0][0], dtype=float)
    results = _som_from_weights(first_W1).predict(norm_df)
    groups = results[results.columns.to_list()[-1]]
    groups = _remap_unseen_groups(groups, found_clusters[0][0], first_W1)

    if mapping == 'godel':
        mapped_groups = map_to_cluster_godel(prev_id=1, layer=1, group=groups + 1)
    else:  # linear or default
        mapped_groups = groups.map(first_layer_mapping)

    new_df = pd.concat([og_df, mapped_groups], axis=1)
    col_list = new_df.columns.to_list()
    new_df = new_df.rename(columns={col_list[-1]: 'layer_1'})

    # --- Secondary layers ------------------------------------------------------
    for i in range(len(LAYERS) - 1):
        layer_name = 'layer_' + str(i + 2)
        new_df[layer_name] = 0.0 if mapping == 'linear' else 0
        prev_layer_name = 'layer_' + str(i + 1)

        flat_child = 0
        for j in range(len(found_clusters[i])):
            for k in range(len(found_clusters[i][j])):
                cluster_id = mapped_found_clusters[i][j][k]
                child_W1 = np.asarray(som_weights[i + 1][flat_child], dtype=float)
                child_stats = min_max[i + 1][flat_child]
                child_observed = found_clusters[i + 1][flat_child]
                flat_child += 1

                rows = new_df[prev_layer_name] == cluster_id
                if not rows.any():
                    continue

                cluster_data = normalize_data(new_df.loc[rows].copy(),
                                              ALL_FEATURES[i + 1],
                                              ALL_SCALES[i + 1],
                                              child_stats,
                                              training=False)
                if len(ALL_FEATURES[i + 1]) == 1:
                    if isinstance(cluster_data, pd.Series):
                        cluster_data = cluster_data.to_frame()
                    cluster_data['aux_col'] = cluster_data[ALL_FEATURES[i + 1][0]]

                cluster_results = _som_from_weights(child_W1).predict(cluster_data)
                cluster_groups = cluster_results[cluster_results.columns.to_list()[-1]]
                cluster_groups = _remap_unseen_groups(cluster_groups,
                                                      child_observed, child_W1)

                lookup = _child_group_lookup(mapping, i, j, k, found_clusters,
                                             mapped_found_clusters,
                                             child_observed, cluster_id)
                new_df.loc[rows, layer_name] = cluster_groups.map(lookup)

    # --- Assemble outputs ----------------------------------------------------
    layer_columns = {}
    n_layers = len(LAYERS)
    for idx in range(n_layers):
        name = 'layer_' + str(idx + 1)
        layer_columns[name] = new_df[name]

    deepest = 'layer_' + str(n_layers)
    cluster_id = new_df[deepest].to_numpy()
    return cluster_id, layer_columns
