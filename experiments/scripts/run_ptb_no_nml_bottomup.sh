#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zihopc/vscode/Jungyeul/20251116_stanza
export PYTHONPATH="$ROOT/stanza-main"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="$ROOT/.venv/bin/python"
DATA_DIR="$ROOT/data/ptb_no_nml"
RUN_DIR="$ROOT/bottomup_runs_ptb_no_nml"
MODEL_DIR="$RUN_DIR/models"
PRED_DIR="$RUN_DIR/predictions"
LOG_DIR="$RUN_DIR/logs"
EVAL_DIR="$RUN_DIR/evalb"

mkdir -p "$MODEL_DIR" "$PRED_DIR" "$LOG_DIR" "$EVAL_DIR"

COMMON_ARGS=(
  --shorthand en_ptb
  --wordvec_pretrain_file /home/zihopc/stanza_resources/en/pretrain/conll17.pt
  --charlm_forward_file /home/zihopc/stanza_resources/en/forward_charlm/1billion.pt
  --charlm_backward_file /home/zihopc/stanza_resources/en/backward_charlm/1billion.pt
  --epochs "${EPOCHS:-100}"
  --epoch_size "${EPOCH_SIZE:-5000}"
  --train_batch_size "${TRAIN_BATCH_SIZE:-16}"
  --eval_batch_size "${EVAL_BATCH_SIZE:-24}"
  --oracle_initial_epoch 999999
  --no_check_valid_states
  --no_retag
  --save_dir "$MODEL_DIR"
  --device "${DEVICE:-cuda}"
)

train_and_predict() {
  local name=$1
  local scheme=$2

  "$PY" -m stanza.models.constituency_parser \
    --train_file "$DATA_DIR/en_ptb_train.mrg" \
    --eval_file "$DATA_DIR/en_ptb_dev.mrg" \
    --mode train \
    --transition_scheme "$scheme" \
    --save_name "${name}.pt" \
    --checkpoint_save_name "${name}_checkpoint.pt" \
    "${COMMON_ARGS[@]}" \
    > "$LOG_DIR/${name}.train.log" 2>&1

  for split in dev test; do
    rm -f "$PRED_DIR/${name}_${split}.pred.mrg" "$PRED_DIR/${name}_${split}.orig.mrg"
    "$PY" -m stanza.models.constituency_parser \
      --eval_file "$DATA_DIR/en_ptb_${split}.mrg" \
      --mode predict \
      --transition_scheme "$scheme" \
      --load_name "$MODEL_DIR/${name}.pt" \
      --predict_dir "$PRED_DIR" \
      --predict_file "${name}_${split}" \
      --no_retag \
      --eval_batch_size "${EVAL_BATCH_SIZE:-24}" \
      --device "${DEVICE:-cuda}" \
      > "$LOG_DIR/${name}.${split}.log" 2>&1

    "$PY" "$ROOT/validate_prediction_yields.py" \
      "$DATA_DIR/en_ptb_${split}.mrg" \
      "$PRED_DIR/${name}_${split}.pred.mrg" \
      > "$LOG_DIR/${name}.${split}.yield.log"

    "$ROOT/run_evalb_official.sh" \
      "$ROOT/evaluators/EVALB/evalb" \
      "$ROOT/evaluators/EVALB/COLLINS.prm" \
      "$DATA_DIR/en_ptb_${split}.mrg" \
      "$PRED_DIR/${name}_${split}.pred.mrg" \
      "$EVAL_DIR/${name}_${split}" \
      > "$LOG_DIR/${name}.${split}.evalb.summary.log"
  done
}

train_and_predict arity BOTTOM_UP_ARITY
train_and_predict delimiter BOTTOM_UP_DELIMITER

echo "PTB no-NML Arity and Delimiter runs completed."
