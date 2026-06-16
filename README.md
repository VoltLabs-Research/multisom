# MultiSOM

Unsupervised structural identification via a multilayer Self-Organizing Map (SOM): clusters atoms into a per-atom `cluster_id` from features produced by selected feature plugins (e.g. coordination-analysis, centrosymmetry-parameter, steinhardt-order-parameters).

## Install

```bash
vpm install @voltlabs/multisom
```

## CLI

```bash
multisom <output_base> --layers '<json>' [options]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `<output_base>` | yes | — | Working/output directory; reads upstream feature parquets here and writes outputs as `{output_base}_<suffix>`. |
| `--layers` | yes | — | JSON array of SOM layers. Each layer: `label`, `features`, `scaling` (`normal`/`standard`/`robust`), `f`, `sigma`, `eta`, `N`, `batched`, `batch_size`. |
| `--training` | no | `reference` | `reference` (stable cluster IDs across frames) or `per_frame` (retrain each frame). |
| `--mapping` | no | `linear` | Cluster ID mapping: `linear`, `godel`, or `default`. |
| `--seed` | no | `1982` | Random seed. |
| `--frameIndex` | no | `0` | Index of the current frame in the filtered trajectory. |

## Exports

| Output file | Exposure | Exporter → artifact |
|---|---|---|
| `{output_base}_atoms.parquet` | MultiSOM Model | AtomisticExporter → glb |
| `{output_base}_multisom.parquet` | MultiSOM | — (listing-only) |

---

Full input contract and examples: https://docs.voltcloud.dev/docs/plugins
