#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zihopc/vscode/Jungyeul/20251116_stanza
export PYTHONPATH="$ROOT/stanza-main"
export CUDA_VISIBLE_DEVICES=0

PY="$ROOT/.venv/bin/python"
RUN_DIR="$ROOT/bottomup_runs_ctb"
MODEL_DIR="$RUN_DIR/models"
PRED_DIR="$RUN_DIR/predictions"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$MODEL_DIR" "$PRED_DIR" "$LOG_DIR"

COMMON_ARGS=(
  --shorthand zh_ctb
  --wordvec_pretrain_file /home/zihopc/stanza_resources/zh-hans/pretrain/fasttext157.pt
  --charlm_forward_file /home/zihopc/stanza_resources/zh-hans/forward_charlm/gigaword.pt
  --charlm_backward_file /home/zihopc/stanza_resources/zh-hans/backward_charlm/gigaword.pt
  --epochs "${EPOCHS:-100}"
  --epoch_size "${EPOCH_SIZE:-5000}"
  --train_batch_size "${TRAIN_BATCH_SIZE:-16}"
  --eval_batch_size "${EVAL_BATCH_SIZE:-24}"
  --oracle_initial_epoch 999999
  --no_check_valid_states
  --no_retag
  --save_dir "$MODEL_DIR"
  --device cuda
)

train_and_predict() {
  local name="$1"
  local scheme="$2"
  local train_file="$3"
  local dev_file="$4"
  local test_file="$5"

  echo "===== TRAIN $name $scheme ====="
  "$PY" -m stanza.models.constituency_parser \
    --train_file "$train_file" \
    --eval_file "$dev_file" \
    --mode train \
    --transition_scheme "$scheme" \
    --save_name "${name}.pt" \
    --checkpoint_save_name "${name}_checkpoint.pt" \
    "${COMMON_ARGS[@]}" \
    > "$LOG_DIR/${name}.train.log" 2>&1

  echo "===== PREDICT DEV $name ====="
  rm -f "$PRED_DIR/${name}_dev.pred.mrg" "$PRED_DIR/${name}_dev.orig.mrg"
  "$PY" -m stanza.models.constituency_parser \
    --eval_file "$dev_file" \
    --mode predict \
    --transition_scheme "$scheme" \
    --load_name "$MODEL_DIR/${name}.pt" \
    --predict_dir "$PRED_DIR" \
    --predict_file "${name}_dev" \
    --no_retag \
    --eval_batch_size "${EVAL_BATCH_SIZE:-24}" \
    --device cuda \
    > "$LOG_DIR/${name}.dev.log" 2>&1

  echo "===== PREDICT TEST $name ====="
  rm -f "$PRED_DIR/${name}_test.pred.mrg" "$PRED_DIR/${name}_test.orig.mrg"
  "$PY" -m stanza.models.constituency_parser \
    --eval_file "$test_file" \
    --mode predict \
    --transition_scheme "$scheme" \
    --load_name "$MODEL_DIR/${name}.pt" \
    --predict_dir "$PRED_DIR" \
    --predict_file "${name}_test" \
    --no_retag \
    --eval_batch_size "${EVAL_BATCH_SIZE:-24}" \
    --device cuda \
    > "$LOG_DIR/${name}.test.log" 2>&1
}

train_and_predict binary BOTTOM_UP_BINARY "$ROOT/data_ctb/CTB_binary/train_zh.txt" "$ROOT/data_ctb/CTB_binary/dev_zh.txt" "$ROOT/data_ctb/CTB_5.1/zh_ctb_test.mrg"
"$PY" "$ROOT/bottomup_runs/debinarize_bottomup.py" "$PRED_DIR/binary_test.pred.mrg" "$PRED_DIR/binary_test.debin.pred.mrg"
"$PY" -m stanza.models.constituency.evaluate_treebanks "$ROOT/data_ctb/CTB_5.1/zh_ctb_test.mrg" "$PRED_DIR/binary_test.debin.pred.mrg" > "$LOG_DIR/binary_test.debin.eval.log" 2>&1

train_and_predict arity BOTTOM_UP_ARITY "$ROOT/data_ctb/CTB_5.1/zh_ctb_train.mrg" "$ROOT/data_ctb/CTB_5.1/zh_ctb_dev.mrg" "$ROOT/data_ctb/CTB_5.1/zh_ctb_test.mrg"

train_and_predict delimiter BOTTOM_UP_DELIMITER "$ROOT/data_ctb/CTB_5.1/zh_ctb_train.mrg" "$ROOT/data_ctb/CTB_5.1/zh_ctb_dev.mrg" "$ROOT/data_ctb/CTB_5.1/zh_ctb_test.mrg"

echo "DONE"
