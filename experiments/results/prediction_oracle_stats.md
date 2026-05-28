# Prediction-Derived Oracle Statistics

These statistics are computed by taking each predicted tree file and extracting the deterministic oracle transition sequence for that predicted tree.

For Binary and Arity, the counts follow the implemented bottom-up oracle and include `FINISH`.

For Delimiter, the counts follow the paper-level oracle abstraction: `SHIFT`, `D-SHIFT`, `NT(X)`, `U-REDUCE`, `REDUCE`, `D-REDUCE`, and `FINISH`. This is different from the compact Stanza implementation representation, where labels are folded into reduce actions.

## Action Inventory Size From Prediction-Derived Oracle

| Corpus | Split | Binary | Arity | Delimiter |
| ------ | ----- | -----: | ----: | --------: |
| PTB    | dev   | 53     | 108   | 77        |
| PTB    | test  | 55     | 119   | 77        |
| CTB    | dev   | 39     | 73    | 58        |
| CTB    | test  | 45     | 65    | 54        |

## Average Oracle Sequence Length From Predictions

| Corpus | Split | Binary | Arity | Delimiter |
| ------ | ----- | -----: | ----: | --------: |
| PTB    | dev   | 48.20  | 44.92 | 113.12    |
| PTB    | test  | 47.92  | 44.83 | 112.98    |
| CTB    | dev   | 39.76  | 42.23 | 102.86    |
| CTB    | test  | 47.02  | 49.89 | 122.22    |

## Operator Frequency From Prediction-Derived Oracle

| Corpus | Split | System | Operator frequencies |
| ------ | ----- | ------ | -------------------- |
| PTB | dev | Binary | SHIFT=40117, REDUCE-BINARY=38417, REDUCE-UNARY=1700, FINISH=1700 |
| PTB | dev | Arity | SHIFT=41219, REDUCE-ARITY=33443, FINISH=1700 |
| PTB | dev | Delimiter | D-SHIFT=22424, NT(X)=74689, U-REDUCE=41219, SHIFT=18795, D-REDUCE=11046, REDUCE=22424, FINISH=1700 |
| PTB | test | Binary | SHIFT=56684, REDUCE-BINARY=54268, REDUCE-UNARY=2416, FINISH=2416 |
| PTB | test | Arity | SHIFT=58236, REDUCE-ARITY=47653, FINISH=2416 |
| PTB | test | Delimiter | D-SHIFT=32272, NT(X)=106153, U-REDUCE=58236, D-REDUCE=15645, SHIFT=25964, REDUCE=32272, FINISH=2416 |
| CTB | dev | Binary | SHIFT=6821, REDUCE-BINARY=6469, REDUCE-UNARY=352, FINISH=352 |
| CTB | dev | Arity | SHIFT=6821, REDUCE-ARITY=7691, FINISH=352 |
| CTB | dev | Delimiter | D-SHIFT=4381, NT(X)=14516, U-REDUCE=6821, D-REDUCE=3314, REDUCE=4381, FINISH=352, SHIFT=2440 |
| CTB | test | Binary | SHIFT=8008, REDUCE-BINARY=7660, REDUCE-UNARY=348, FINISH=348 |
| CTB | test | Arity | SHIFT=8008, REDUCE-ARITY=9005, FINISH=348 |
| CTB | test | Delimiter | D-SHIFT=5078, NT(X)=17089, U-REDUCE=8008, SHIFT=2930, D-REDUCE=4003, REDUCE=5078, FINISH=348 |
