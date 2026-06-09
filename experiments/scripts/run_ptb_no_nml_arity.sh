#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/zihopc/vscode/Jungyeul/20251116_stanza}
PY=${PY:-$ROOT/.venv/bin/python}
RESOURCE_ROOT=${RESOURCE_ROOT:-$HOME/stanza_resources}
IMPLEMENTATION_ROOT=${IMPLEMENTATION_ROOT:-$ROOT}
export CORENLP_HOME=${CORENLP_HOME:-$HOME/stanford-corenlp-4.5.10}
SOURCE_DIR=$ROOT/data
DATA_DIR=$ROOT/data/ptb_no_nml
RUN_DIR=${RUN_DIR:-$ROOT/bottomup_runs_ptb_no_nml_arity}
MODEL_DIR=$RUN_DIR/models
PRED_DIR=$RUN_DIR/predictions
LOG_DIR=$RUN_DIR/logs
EVAL_DIR=$RUN_DIR/evalb
META_DIR=$RUN_DIR/metadata
PHASE=${1:-all}

export PYTHONPATH="$ROOT/stanza-main"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$DATA_DIR" "$MODEL_DIR" "$PRED_DIR" "$LOG_DIR" "$EVAL_DIR" "$META_DIR"

prepare_data() {
  "$PY" "$ROOT/bottomup_runs/debinarize_bottomup.py" \
    "$SOURCE_DIR/train.txt" "$DATA_DIR/en_ptb_train.mrg"
  "$PY" "$ROOT/bottomup_runs/debinarize_bottomup.py" \
    "$SOURCE_DIR/dev.txt" "$DATA_DIR/en_ptb_dev.mrg"
  cp "$SOURCE_DIR/en_ptb_test.mrg" "$DATA_DIR/en_ptb_test.mrg"

  for file in "$DATA_DIR"/en_ptb_{train,dev,test}.mrg; do
    if grep -Eq '\((NML|MLL)([ (])' "$file"; then
      echo "Unexpected NML/MLL label in $file" >&2
      exit 1
    fi
  done

  "$PY" "$ROOT/validate_prediction_yields.py" \
    "$SOURCE_DIR/train.txt" "$DATA_DIR/en_ptb_train.mrg" \
    > "$LOG_DIR/train_data_yield.log"
  "$PY" "$ROOT/validate_prediction_yields.py" \
    "$SOURCE_DIR/dev.txt" "$DATA_DIR/en_ptb_dev.mrg" \
    > "$LOG_DIR/dev_data_yield.log"

  {
    date --iso-8601=seconds
    uname -a
    "$PY" -c 'import platform, stanza, sys, torch; print("python", sys.version.replace("\n", " ")); print("torch", torch.__version__); print("stanza", stanza.__version__); print("platform", platform.platform()); print("cuda_available", torch.cuda.is_available())'
    if git -C "$IMPLEMENTATION_ROOT" rev-parse HEAD 2>/dev/null; then
      :
    elif [[ -f "$ROOT/IMPLEMENTATION_COMMIT.txt" ]]; then
      printf "git_commit "
      cat "$ROOT/IMPLEMENTATION_COMMIT.txt"
    else
      echo "git_commit unavailable"
    fi
    sha256sum \
      "$SOURCE_DIR/train.txt" \
      "$SOURCE_DIR/dev.txt" \
      "$SOURCE_DIR/en_ptb_test.mrg" \
      "$DATA_DIR/en_ptb_train.mrg" \
      "$DATA_DIR/en_ptb_dev.mrg" \
      "$DATA_DIR/en_ptb_test.mrg"
  } > "$META_DIR/environment_and_data.txt"
}

train() {
  "$PY" -m stanza.models.constituency_parser \
    --train_file "$DATA_DIR/en_ptb_train.mrg" \
    --eval_file "$DATA_DIR/en_ptb_dev.mrg" \
    --mode train \
    --transition_scheme BOTTOM_UP_ARITY \
    --shorthand en_ptb \
    --wordvec_pretrain_file "$RESOURCE_ROOT/en/pretrain/conll17.pt" \
    --charlm_forward_file "$RESOURCE_ROOT/en/forward_charlm/1billion.pt" \
    --charlm_backward_file "$RESOURCE_ROOT/en/backward_charlm/1billion.pt" \
    --epochs "${EPOCHS:-100}" \
    --epoch_size "${EPOCH_SIZE:-5000}" \
    --train_batch_size "${TRAIN_BATCH_SIZE:-16}" \
    --eval_batch_size "${EVAL_BATCH_SIZE:-24}" \
    --oracle_initial_epoch 999999 \
    --no_check_valid_states \
    --no_retag \
    --seed "${SEED:-1234}" \
    --save_dir "$MODEL_DIR" \
    --save_name arity.pt \
    --checkpoint_save_name arity_checkpoint.pt \
    --device "${DEVICE:-cuda}" \
    > "$LOG_DIR/arity.train.log" 2>&1
}

predict_and_eval() {
  local split=$1
  local prefix=arity_${split}

  rm -f "$PRED_DIR/${prefix}.pred.mrg" "$PRED_DIR/${prefix}.orig.mrg"
  "$PY" -m stanza.models.constituency_parser \
    --eval_file "$DATA_DIR/en_ptb_${split}.mrg" \
    --mode predict \
    --transition_scheme BOTTOM_UP_ARITY \
    --load_name "$MODEL_DIR/arity.pt" \
    --predict_dir "$PRED_DIR" \
    --predict_file "$prefix" \
    --no_retag \
    --seed "${SEED:-1234}" \
    --eval_batch_size "${EVAL_BATCH_SIZE:-24}" \
    --device "${DEVICE:-cuda}" \
    > "$LOG_DIR/arity.${split}.log" 2>&1

  "$PY" "$ROOT/validate_prediction_yields.py" \
    "$DATA_DIR/en_ptb_${split}.mrg" \
    "$PRED_DIR/${prefix}.pred.mrg" \
    > "$LOG_DIR/arity.${split}.yield.log"

  "$ROOT/run_evalb_official.sh" \
    "$ROOT/evaluators/EVALB/evalb" \
    "$ROOT/evaluators/EVALB/COLLINS.prm" \
    "$DATA_DIR/en_ptb_${split}.mrg" \
    "$PRED_DIR/${prefix}.pred.mrg" \
    "$EVAL_DIR/${prefix}" \
    > "$LOG_DIR/arity.${split}.evalb.summary.log"
}

case "$PHASE" in
  prepare)
    prepare_data
    ;;
  train)
    prepare_data
    train
    ;;
  eval)
    predict_and_eval dev
    predict_and_eval test
    ;;
  all)
    prepare_data
    train
    predict_and_eval dev
    predict_and_eval test
    ;;
  *)
    echo "Usage: $0 {prepare|train|eval|all}" >&2
    exit 2
    ;;
esac
