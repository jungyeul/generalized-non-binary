# Bottom-up Tables 2-6

Table 2 and Table 4 use training gold oracle sequences. Table 5 and Table 6 use test gold/predicted trees. Root/TOP nodes and preterminals are excluded from Table 5 subtree arity counts. Prediction yields are validated against gold before evaluation.

## Table 2: Average transition sequence length per sentence
| System                     | PTB   | CTB   |
| -------------------------- | ----- | ----- |
| Binary bottom-up           | 47.70 | 54.64 |
| Arity-specific bottom-up   | 44.70 | 59.45 |
| Delimiter-guided bottom-up | 58.31 | 77.56 |

## Table 4: Action inventory size used in gold oracle sequences
| System                     | PTB full | CTB full | PTB reductions | CTB reductions |
| -------------------------- | -------- | -------- | -------------- | -------------- |
| Binary bottom-up           | 125      | 147      | 124            | 146            |
| Arity-specific bottom-up   | 204      | 189      | 203            | 188            |
| Delimiter-guided bottom-up | 54       | 50       | 52             | 48             |

## Table 5: PTB subtree arity distribution
| Source | k=1  | k=2   | k=3  | k=4  | k>=5 |
| ------ | ---- | ----- | ---- | ---- | ---- |
| G      | 9012 | 24440 | 8214 | 2342 | 1694 |
| A      | 8875 | 24389 | 8262 | 2296 | 1707 |
| D      | 8901 | 24296 | 8242 | 2357 | 1705 |

## Table 5: CTB subtree arity distribution
| Source | k=1  | k=2  | k=3  | k=4 | k>=5 |
| ------ | ---- | ---- | ---- | --- | ---- |
| G      | 3482 | 3698 | 1121 | 184 | 235  |
| A      | 3414 | 3712 | 1117 | 183 | 231  |
| D      | 3470 | 3748 | 1099 | 187 | 229  |

## Table 6: Boundary prediction errors in delimiter-guided parser
| Error type        | PTB | CTB |
| ----------------- | --- | --- |
| Missing D-SHIFT   | 381 | 86  |
| Spurious D-SHIFT  | 314 | 94  |
| Premature REDUCE  | 506 | 125 |
| Spurious D-REDUCE | 495 | 108 |

Diagnostics:
- PTB Table 6 skipped 0 sentence(s) due to delimiter prediction leaf mismatches.
- CTB Table 6 skipped 0 sentence(s) due to delimiter prediction leaf mismatches.