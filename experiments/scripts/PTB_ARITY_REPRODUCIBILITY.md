# PTB Arity Reproduction

This pipeline trains only the Arity-specific bottom-up parser on the unified
PTB representation without `NML` or `MLL` labels, then evaluates dev and test
with the original NYU EVALB implementation.

## Run

```bash
cd /home/zihopc/vscode/Jungyeul/20251116_stanza
DEVICE=cuda ./run_ptb_no_nml_arity.sh all
```

For another installation location:

```bash
ROOT="$HOME/ziho_ptb_arity" \
PY="$HOME/venvs/ziho/bin/python" \
RESOURCE_ROOT="$HOME/stanza_resources" \
CORENLP_HOME="$HOME/stanford-corenlp-4.5.10" \
DEVICE=cuda \
"$HOME/ziho_ptb_arity/run_ptb_no_nml_arity.sh" all
```

The random seed is fixed to `1234`. To resume after interruption, run the same
command. Stanza will load `arity_checkpoint.pt` when it is present.

CPU fallback:

```bash
DEVICE=cpu ./run_ptb_no_nml_arity.sh all
```

## Data

The non-binary train and dev files are deterministically debinarized from:

```text
data/train.txt
data/dev.txt
```

The test file is:

```text
data/en_ptb_test.mrg
```

Generated files are stored under:

```text
data/ptb_no_nml/
```

The pipeline aborts if `NML` or `MLL` occurs, or if debinarization changes any
sentence yield.

## EVALB

The pipeline uses:

```bash
evaluators/EVALB/evalb \
  -p evaluators/EVALB/COLLINS.prm \
  -e 10000 \
  GOLD.mrg \
  PRED.mrg
```

`-e 10000` must follow `-p`; otherwise `MAX_ERROR 10` in `COLLINS.prm`
overwrites the command-line setting.

## Outputs

All outputs are under:

```text
bottomup_runs_ptb_no_nml_arity/
```

- `models/arity.pt`: best model
- `models/arity_checkpoint.pt`: restart checkpoint
- `predictions/`: dev and test predictions
- `evalb/`: complete original EVALB output and error files
- `logs/`: training, prediction, yield-validation, and EVALB summaries
- `metadata/environment_and_data.txt`: runtime versions, Git commit, and data checksums
