# Delimiter-guided non-binary constituency parsing

Experiment code and results for **A generalized transition framework for identifying reduction domains in non-binary constituency parsing** by Jungyeul Park, Zihao Huang, and Eunkyul Leah Jo.

| Traversal order | Canonical | Delimiter-guided |
| --- | --- | --- |
| Top-down | [Notebook](notebooks/canonical_TD.ipynb) | [Notebook](notebooks/generalised_TD.ipynb) |
| In-order | [Notebook](notebooks/canonical_IO.ipynb) | [Notebook](notebooks/generalised_IO.ipynb) |
| Bottom-up | [Notebook](notebooks/canonical_BU.ipynb) | [Notebook](notebooks/generalised_BU.ipynb) |

The notebooks demonstrate static oracles and exact reconstruction with synthetic trees. The two bundled Stanza source trees preserve the implementations used for the canonical and delimiter-guided experiments. Run the commands below from the repository root.

## Set up the environment

Use Linux with a CUDA-capable GPU for the paper's training protocol. Install a PyTorch build compatible with your CUDA setup and make Java available for CoreNLP. The archived environment used Python 3.10.12, PyTorch 2.14.1+cu130, Transformers 4.57.6, Stanza 1.14.0, and CoreNLP 4.6.0.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./stanza-main-constituency-generalised 'transformers==4.57.6' nltk
```

The launcher selects the appropriate source tree for each run. The first training run downloads the configured BERT and Stanza resources if they are absent. For interactive notebooks, install JupyterLab in the same environment and select that Python kernel.

## Prepare your licensed treebanks

Obtain [Penn Treebank 3 (LDC99T42)](https://catalog.ldc.upenn.edu/LDC99T42) and the [Chinese Treebank 5.1 update to LDC2005T01](https://catalog.ldc.upenn.edu/LDC2005T01). Provide these prepared files, one bracketed tree per line.

```text
data/PTB_3/en_ptb_{train,dev,test}.mrg
data/CTB_5.1/zh_ctb_{train,dev,test}.mrg
```

For raw PTB, pass `--ptb-dir /path/to/wsj` to the launcher. It uses sections 02–21 for training, 22 for development, and 23 for testing. For raw CTB 5.1, use the included Stanza converter.

```bash
python scripts/prepare_ctb.py /path/to/CTB5.1/bracketed data/CTB_5.1
```

The CTB basic split uses documents 001–270 and 440–1151 for training, 301–325 for development, and 271–300 for testing. The runner normalizes the trees before training. The archive contains 39,832/1,700/2,416 PTB trees and 17,544/352/348 CTB trees before train and development deduplication. Compare the normalized files against the SHA-256 fingerprints in the [archived run provenance](results/archived-run-provenance.json).

## Train and evaluate

```bash
python scripts/run_experiments.py --data-root data --device cuda:0 --dry-run
python scripts/run_experiments.py --data-root data --device cuda:0 \
  --corenlp-home /path/to/stanford-corenlp-4.6.0
```

For automatic CoreNLP installation, pass `--install-corenlp`. The default runs all 36 configurations sequentially. Each parser and corpus uses seeds 1111, 2222, and 3333, and selects each seed's checkpoint by development F1. Use `--families`, `--orders`, and `--corpora` to select subsets.

```bash
python scripts/run_experiments.py --data-root data --families generalised \
  --orders bu --corpora ctb --device cuda:0 --corenlp-home /path/to/stanford-corenlp-4.6.0
```

Checkpoints, run manifests, logs, and predictions go to the corresponding checkout under `saved_models/constituency/<family>_<order>_static/<corpus>/seed_<seed>/`. The parent directory contains a summary across seeds.

To run three seeds concurrently on three GPUs, use the bundled schedulers with prepared treebanks. For each parser, the scheduler aggregates all PTB seeds before starting CTB. Replace `0 1 2` with your physical GPU indices for seeds 1111, 2222, and 3333 in that order. Add `--dry-run` to inspect the scheduled runs.

```bash
for family in canonical generalised; do
  checkout=stanza-main-constituency
  if [ "$family" = generalised ]; then
    checkout=stanza-main-constituency-generalised
  fi
  for order in td io bu; do
    python "$checkout/scripts/run_${family}_${order}_experiments.py" \
      --data-root data --gpus 0 1 2 --corenlp-home /path/to/stanford-corenlp-4.6.0 || exit
  done
done
```

The archived canonical top-down PTB runs resumed after interruptions. The launcher starts fresh runs by default. An exact replay of those runs would also need their interrupted checkpoints and resume schedule.

### Reproduce the reported scores

The native runner uses CoreNLP for checkpoint selection. To reproduce the paper's reported test metrics, install the two pinned scorers and rescore the exported trees.

```bash
mkdir -p scorers
git clone https://github.com/shlomihod/Evalb.git scorers/EVALB
git -C scorers/EVALB checkout 00c15be0abea14f2bff03ef2007f2e0cf6f2f5a3
make -B -C scorers/EVALB
git clone https://github.com/jungyeul/alignment-based-PARSEVAL.git scorers/alignment-based-PARSEVAL
git -C scorers/alignment-based-PARSEVAL checkout f82c4fc0e08fc5f597f379aeedcaee377b57495e
python scripts/rescore.py --scorers-dir scorers --output outputs/rescored
```

The rescorer uses EVALB with the `COLLINS.prm` distributed with JP-EVALB and JP-EVALB's default alignment mode. It reports per-seed corpus F1, mean, and sample standard deviation. Use a new output directory for each rescoring run.

### Decoding benchmark and transition audit

The timing script requires Linux CPU affinity and cached model resources. Choose an available physical CPU core and run the benchmark for each corpus and seed. Each invocation times all six parsers using the same test sentences for that corpus. Install Matplotlib to regenerate the plot.

```bash
python scripts/benchmark_decoding.py --work . --resources ~/stanza_resources \
  --output outputs/timing/ptb-1111 --cpu 2 --corpus ptb --seed 1111
python scripts/plot_decoding.py results/decoding-benchmark-results.json outputs/latency.png
```

For new timings, name the block directories `ptb-1111` through `ctb-3333` as above. Run `python scripts/summarize_decoding.py outputs/timing outputs/timing-summary.json` and pass the new JSON to the plot script.

For transition conformance, collect one trace set per family, corpus, and seed, then audit the combined directory.

```bash
for family in canonical generalised; do
  for corpus in ptb ctb; do
    for seed in 1111 2222 3333; do
      python scripts/trace_decoding.py --work . --resources ~/stanza_resources \
        --output outputs/traces --family "$family" --corpus "$corpus" --seed "$seed"
    done
  done
done
python scripts/audit_decoding.py outputs/traces outputs/audit.json
```

Trace files contain reconstructed trees over licensed tokens. Keep them local along with your corpora and exported predictions.

## Archived results

Test F1 is reported as mean ± sample standard deviation over three seeds.

| System | PTB EVALB | PTB JP-EVALB | CTB EVALB | CTB JP-EVALB |
| --- | ---: | ---: | ---: | ---: |
| Canonical top-down | 94.86 ± 0.07 | 94.27 ± 0.09 | 89.83 ± 0.28 | 88.09 ± 0.31 |
| Canonical in-order | 95.04 ± 0.07 | 94.35 ± 0.08 | 90.62 ± 0.19 | 88.89 ± 0.20 |
| Canonical bottom-up | 94.90 ± 0.04 | 94.27 ± 0.09 | 90.63 ± 0.27 | 88.89 ± 0.26 |
| Delimiter-guided top-down | 94.89 ± 0.10 | 94.25 ± 0.11 | 89.90 ± 0.33 | 88.20 ± 0.35 |
| Delimiter-guided in-order | 94.84 ± 0.05 | 94.18 ± 0.04 | 90.48 ± 0.38 | 88.74 ± 0.34 |
| Delimiter-guided bottom-up | 94.90 ± 0.02 | 94.21 ± 0.04 | 90.94 ± 0.33 | 89.20 ± 0.33 |

- [Per-seed test accuracy](results/per-seed-accuracy.csv), [mean and sample SD](results/accuracy-summary.csv), and [full score and bracket-count records](results/accuracy.json).
- [Decoding benchmark](results/decoding-benchmark-results.json), [latency figure](results/decoding-latency.png), and [transition conformance audit](results/decoding-rule-audit-results.json).
- [Structural and workload analyses](results/analysis-results.json), [arity analysis](results/arity-results.json), and [archived run provenance](results/archived-run-provenance.json).

The vendored Stanza sources retain the upstream Apache 2.0 license in each checkout. Runtime files outside the experiment changes match the public [Stanza v1.14.0 release](https://github.com/stanfordnlp/stanza/releases/tag/v1.14.0). The archived provenance maps the recorded source hashes to the packaged files. The historical launcher variants used by delimiter-guided bottom-up runs are in that checkout's `archived-launchers/` directory. Obtain both corpora from the LDC under its terms.

Related work by the authors is [Non-binary bottom-up constituency parsing without arity actions](https://aclanthology.org/2026.findings-emnlp.611/).
