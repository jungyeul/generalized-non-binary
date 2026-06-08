#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 EVALB_BIN PARAM_FILE GOLD_FILE PRED_FILE OUTPUT_PREFIX" >&2
  exit 2
fi

evalb_bin=$1
param_file=$2
gold_file=$3
pred_file=$4
output_prefix=$5

mkdir -p "$(dirname "$output_prefix")"

# -e must follow -p because MAX_ERROR in the parameter file otherwise
# overwrites the command-line value.
"$evalb_bin" \
  -p "$param_file" \
  -e 10000 \
  "$gold_file" \
  "$pred_file" \
  > "${output_prefix}.txt" \
  2> "${output_prefix}.err"

grep -A 16 '^-- All --' "${output_prefix}.txt"

if [[ -s "${output_prefix}.err" ]]; then
  echo "EVALB reported yield/token errors; see ${output_prefix}.err" >&2
  exit 1
fi
