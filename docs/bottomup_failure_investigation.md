# Binary and PTB Arity Failure Investigation

Updated: 2026-06-08

This note summarizes the current investigation into two low-priority but important issues:

- poor binary test results
- the historical 24 invalid PTB arity test predictions

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

## PTB Arity Repeated Sentence

The original PTB arity test prediction contained 24 invalid ROOT/yield cases,
but only one sentence actually reached the decoder transition limit.

Observed pattern:

- All 24 have non-unary ROOT: `root-arity=20`.
- 23 / 24 also have a yield mismatch.
- All 24 invalid outputs are the same flat sentence:

```text
( See related story : `` Fed Ready to Inject Big Funds '' -- WSJ Oct. 16 , 1989 )
```

Sentence 84 (1-based; index 83) is the only case where that yield actually
matches the gold sentence, but it still had an invalid non-unary ROOT. The
other 23 cases incorrectly reused this same output for unrelated 17-token
sentences.

The root cause was in `BaseModel.bulk_apply`. When one state was marked
`broken`, transitions were filtered but the corresponding state list was not.
The subsequent `zip` paired transitions with the wrong states, copying the
broken sentence's constituent stack into the remaining active batch entries.

The fix filters states and transitions together. It also emits a legal unary
`ROOT` fallback for the one genuinely broken decode and verifies that parse
result indices are complete and unique.

Verified corrected result:

- prediction trees: 2416
- yield mismatches: 0
- original EVALB error sentences: 0
- Stanza F1: 92.8478 on N=2416
- original EVALB F1: 93.12 on N=2416

The remaining decoder issue is one sentence that repeatedly predicts unary
`FRAG` reductions until the transition cap. It is now isolated to that sentence
and counted through the valid-yield fallback instead of corrupting other
predictions.

## Original EVALB

`COLLINS.prm` contains `MAX_ERROR 10`. EVALB processes options in order, so
`-e 10000 -p COLLINS.prm` silently resets the limit back to 10. The correct
order is:

```bash
evalb -p COLLINS.prm -e 10000 GOLD PRED
```

The repository wrapper `experiments/scripts/run_evalb_official.sh` uses this
order and fails if EVALB writes any token/yield errors.

## Punctuation Audit

All current Arity and Delimiter dev/test predictions for PTB and CTB preserve
the gold yield exactly. The punctuation problem is isolated to the binary PTB
data and the old corrupted Arity output.

The supplied PTB binary files change tokenization, for example:

- `500 - stock` becomes `500-stock`
- `Capital Cities / ABC` becomes `Capital Cities\\/ABC`
- `finger - pointing` becomes `finger-pointing`

This cannot be repaired reliably by EVALB or by structural debinarization.
Binary PTB test input must be generated with the same transformation and must
retain a reversible token map back to the original PTB test yield.

## Diagnostic Scripts

The scripts used for this investigation are:

- `experiments/scripts/diagnose_bottomup_failures.py`
- `experiments/scripts/binary_structure_diagnosis.py`
- `experiments/scripts/compare_preterminals.py`

They are diagnostic only and do not modify model outputs.
