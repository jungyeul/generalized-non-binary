# Binary and PTB Arity Failure Investigation

Date: 2026-05-27

This note summarizes the current investigation into two low-priority but important issues:

- poor binary test results
- 24 invalid PTB arity test predictions

## Binary Results

### PTB binary

The PTB binary predictions have many yield mismatches against the original PTB gold files.

Observed counts:

- PTB binary dev: 415 / 1700 sentences have leaf/yield mismatches.
- PTB binary test: 599 / 2416 sentences have leaf/yield mismatches.
- Debinarization removes temporary labels, but it does not fix yield mismatches.

Typical examples show tokenization differences:

- gold: `500 - stock`
- binary prediction: `500-stock`

and:

- gold: `Capital Cities / ABC`
- binary prediction: `Capital Cities\\/ABC`

This means the binary PTB files are not only structurally binarized; they also use a tokenization that differs from the original PTB evaluation files. Any EVALB-style score against the original PTB gold is therefore penalized heavily or skips/invalidates many sentences.

Likely fix before rebuttal:

1. Generate or obtain a binary PTB test file with the exact same tokenization/transformation as `PTB_binary/train.txt` and `PTB_binary/dev.txt`.
2. Predict on that binary test input.
3. Debinarize back to the original PTB tokenization, or explicitly map merged tokens back before evaluation.

### CTB binary

CTB binary has no yield mismatch after debinarization:

- CTB binary dev: 0 leaf mismatches.
- CTB binary test: 0 leaf mismatches.

However, the CTB binary test structure is much worse than arity/delimiter on the same test set:

- CTB binary test debinarized: F1 60.97
- CTB arity test: F1 91.09
- CTB delimiter test: F1 90.72

The structural diagnostic shows that CTB binary test recall collapses:

- matched / predicted / gold = 4373 / 5625 / 8720
- precision = 77.74
- recall = 50.15
- F1 = 60.97

The likely cause is an input representation mismatch. The binary CTB dev file has transformed preterminal labels such as `NP+NN`, `NP+NR`, `CLP+M`, `ADVP+AD`, etc. The original CTB test file has normal POS labels such as `NN`, `NR`, `VV`, `PU`, etc.

Since prediction was run with `--no_retag`, the parser used the tags from the eval file. Binary dev was predicted with binary-transformed tags, but binary test was predicted from the original CTB test file with original tags. This creates a train/dev-vs-test mismatch for the binary system.

Likely fix before rebuttal:

1. Generate a CTB binary test file using the same binary transformation as `CTB_binary/train_zh.txt` and `CTB_binary/dev_zh.txt`.
2. Predict binary test from that binary-transformed test file.
3. Debinarize and evaluate against the original CTB test file.

## PTB Arity 24 Failed Sentences

The raw PTB arity test prediction contains 24 invalid ROOT/yield cases.

Observed pattern:

- All 24 have non-unary ROOT: `root-arity=20`.
- 23 / 24 also have a yield mismatch.
- All 24 invalid outputs are the same flat sentence:

```text
( See related story : `` Fed Ready to Inject Big Funds '' -- WSJ Oct. 16 , 1989 )
```

Sentence 83 is the only case where that yield actually matches the gold sentence, but it still has an invalid non-unary ROOT. The other 23 cases incorrectly reuse this same output for unrelated 17-token sentences.

This pattern does not look like ordinary model parsing error. It is more consistent with an output bookkeeping issue during batch prediction: one predicted tree is being reused/written for multiple sentence positions after decoding. The earlier `base_model.py` batch-index patch was audited but did not eliminate the issue, so the next step is to instrument prediction with sentence IDs and verify where the output tree is assigned to the result list.

For the paper numbers, we used the conservative valid-yield fallback file so all 2416 PTB test sentences are counted:

- raw Stanza arity test: F1 92.84 on N=2392
- conservative fallback: F1 92.53 on N=2416

## Diagnostic Scripts

The scripts used for this investigation are:

- `experiments/scripts/diagnose_bottomup_failures.py`
- `experiments/scripts/binary_structure_diagnosis.py`
- `experiments/scripts/compare_preterminals.py`

They are diagnostic only and do not modify model outputs.
