# Bottom-up Stanza Implementation

This branch contains the current Stanza-based implementation used for the bottom-up constituency parsing experiments.

## Contents

- `stanza-main/`: modified Stanza source tree. The bottom-up constituency parser changes are under `stanza-main/stanza/models/constituency/`.
- `experiments/scripts/`: scripts used to run PTB/CTB experiments, debinarize binary outputs, extract oracle statistics, and compute arity-specific F-scores.
- `experiments/results/`: compact result tables and diagnostic summaries generated from the completed PTB/CTB runs.
- `docs/base_model_batch_index_fix.diff`: audit patch used while investigating invalid-yield PTB arity predictions.

Large generated files are intentionally not committed: treebank data, model checkpoints, training logs, evaluation logs, and full prediction/oracle dumps.

## Stanza Source Reference

The implementation was built on a local copy of Stanza version `1.11.0`, as recorded in `stanza-main/stanza/_version.py`.

For future upstream merging reference, the local source directory on the Linux server was already present by `2026-01-03 20:58:31 UTC` (`/home/zihopc/vscode/Jungyeul/20251116_stanza/stanza-main`). I do not currently have the original upstream Git commit hash because that local source copy was not preserved as a Git checkout.

## Main Implementation Files

The core bottom-up changes are concentrated in:

- `stanza-main/stanza/models/constituency/parse_transitions.py`
- `stanza-main/stanza/models/constituency/transition_sequence.py`
- `stanza-main/stanza/models/constituency/utils.py`
- `stanza-main/stanza/models/constituency/evaluate_treebanks.py`
- `stanza-main/stanza/models/constituency/parser_training.py`
- `stanza-main/stanza/models/constituency/base_model.py`

The implementation adds and exercises three bottom-up transition systems:

- binary bottom-up over binarized trees
- arity-specific non-binary bottom-up
- delimiter-guided non-binary bottom-up

## Notes

PTB arity test predictions had 24 invalid-yield outputs in the raw prediction file. The reported full-test arity numbers use the conservative valid-yield fallback prediction so that all 2416 PTB test sentences are counted.
