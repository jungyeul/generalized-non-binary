#!/usr/bin/env python3
import json
from pathlib import Path

from stanza.models.constituency import tree_reader, transition_sequence
from stanza.models.constituency.parse_transitions import (
    Shift,
    TransitionScheme,
    BottomUpUnary,
    BottomUpBinary,
    BottomUpArity,
)


ROOT = Path("/home/zihopc/vscode/Jungyeul/20251116_stanza")
OUT = ROOT / "prediction_oracles"

FILES = {
    ("PTB", "dev", "Binary"): ROOT / "bottomup_runs/predictions/binary_dev.pred.mrg",
    ("PTB", "test", "Binary"): ROOT / "bottomup_runs/predictions/binary_test.pred.mrg",
    ("PTB", "dev", "Arity"): ROOT / "bottomup_runs/predictions/arity_dev.pred.mrg",
    ("PTB", "test", "Arity"): ROOT / "bottomup_runs/predictions/arity_test.validyield_fallback.pred.mrg",
    ("PTB", "dev", "Delimiter"): ROOT / "bottomup_runs/predictions/delimiter_dev.pred.mrg",
    ("PTB", "test", "Delimiter"): ROOT / "bottomup_runs/predictions/delimiter_test.pred.mrg",
    ("CTB", "dev", "Binary"): ROOT / "bottomup_runs_ctb/predictions/binary_dev.pred.mrg",
    ("CTB", "test", "Binary"): ROOT / "bottomup_runs_ctb/predictions/binary_test.pred.mrg",
    ("CTB", "dev", "Arity"): ROOT / "bottomup_runs_ctb/predictions/arity_dev.pred.mrg",
    ("CTB", "test", "Arity"): ROOT / "bottomup_runs_ctb/predictions/arity_test.pred.mrg",
    ("CTB", "dev", "Delimiter"): ROOT / "bottomup_runs_ctb/predictions/delimiter_dev.pred.mrg",
    ("CTB", "test", "Delimiter"): ROOT / "bottomup_runs_ctb/predictions/delimiter_test.pred.mrg",
}

SCHEMES = {
    "Binary": TransitionScheme.BOTTOM_UP_BINARY,
    "Arity": TransitionScheme.BOTTOM_UP_ARITY,
    "Delimiter": TransitionScheme.BOTTOM_UP_DELIMITER,
}


def label_text(label):
    if isinstance(label, (tuple, list)):
        return "+".join(label)
    return str(label)


def paper_action(action):
    if isinstance(action, Shift):
        return "SHIFT"
    if isinstance(action, BottomUpUnary):
        return f"REDUCE-UNARY-{label_text(action.label)}"
    if isinstance(action, BottomUpBinary):
        return f"REDUCE-{action.direction}-{label_text(action.label)}"
    if isinstance(action, BottomUpArity):
        return f"REDUCE-{label_text(action.label)}#{action.arity}"
    return repr(action)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_implementation_oracle(corpus, split, system, trees):
    seqs = transition_sequence.build_treebank(trees, SCHEMES[system])
    rows = []
    lines = []
    paper_lines = []
    for idx, seq in enumerate(seqs):
        actions = [repr(action) for action in seq] + ["FINISH"]
        paper_actions = [paper_action(action) for action in seq] + ["FINISH"]
        rows.append({"sent_id": idx, "actions": actions, "paper_actions": paper_actions})
        lines.append(" ".join(actions))
        paper_lines.append(" ".join(paper_actions))

    out_dir = OUT / "implementation" / corpus.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{corpus.lower()}_{split}_{system.lower()}"
    (out_dir / f"{stem}.oracle.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / f"{stem}.paper_actions.txt").write_text("\n".join(paper_lines) + "\n", encoding="utf-8")
    write_jsonl(out_dir / f"{stem}.oracle.jsonl", rows)


def delimiter_paper_oracle_for_tree(tree):
    parent_of = {}
    child_index_of = {}

    def register(node):
        for idx, child in enumerate(node.children):
            parent_of[id(child)] = node
            child_index_of[id(child)] = idx
            register(child)

    register(tree)

    def highest_left_edge_phrasal_ancestor(preterminal):
        current = preterminal
        target = None
        while id(current) in parent_of:
            parent = parent_of[id(current)]
            child_index = child_index_of[id(current)]
            if child_index != 0:
                break
            if not parent.is_preterminal():
                target = parent
            current = parent
        return target

    actions = []
    previous_parent_id = None
    terminal_index = 0

    def shift_word(parent, preterminal):
        nonlocal previous_parent_id, terminal_index
        current_parent_id = id(parent) if parent is not None else None
        is_first_terminal = terminal_index == 0
        parent_changed = previous_parent_id is not None and current_parent_id != previous_parent_id
        target = highest_left_edge_phrasal_ancestor(preterminal) if preterminal is not None else None
        use_d_shift = target is not None and (is_first_terminal or parent_changed)
        if use_d_shift:
            actions.append("D-SHIFT")
            boundary = id(target)
        else:
            actions.append("SHIFT")
            boundary = None
        if preterminal is not None:
            actions.append(f"NT({preterminal.label})")
            actions.append("U-REDUCE")
        start = terminal_index
        terminal_index += 1
        previous_parent_id = current_parent_id
        return boundary, start, terminal_index

    def visit(node, parent=None):
        if node.is_leaf():
            return shift_word(parent, None)
        if node.is_preterminal():
            return shift_word(parent, node)

        child_boundaries = []
        child_spans = []
        for child in node.children:
            boundary, start, end = visit(child, node)
            child_boundaries.append(boundary)
            child_spans.append((start, end))

        actions.append(f"NT({node.label})")
        active_boundaries = [boundary for boundary in child_boundaries if boundary is not None]
        if active_boundaries and not (len(active_boundaries) == 1 and active_boundaries[0] == id(node)):
            actions.append("D-REDUCE")
            boundary = active_boundaries[0]
        else:
            actions.append("REDUCE")
            boundary = None
        return boundary, child_spans[0][0], child_spans[-1][1]

    visit(tree)
    actions.append("FINISH")
    return actions


def dump_delimiter_paper_oracle(corpus, split, trees):
    rows = []
    lines = []
    for idx, tree in enumerate(trees):
        actions = delimiter_paper_oracle_for_tree(tree)
        rows.append({"sent_id": idx, "actions": actions})
        lines.append(" ".join(actions))

    out_dir = OUT / "paper_abstraction" / corpus.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{corpus.lower()}_{split}_delimiter"
    (out_dir / f"{stem}.oracle.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_jsonl(out_dir / f"{stem}.oracle.jsonl", rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    readme = [
        "Prediction-derived oracle files",
        "",
        "Each file is produced by reading a predicted tree file and extracting a deterministic oracle sequence from that predicted tree.",
        "",
        "implementation/: exact Stanza implementation action reprs, plus FINISH appended.",
        "implementation/*.paper_actions.txt: same Stanza sequences rendered in simpler paper-style names.",
        "paper_abstraction/: delimiter oracle expanded to the paper abstraction SHIFT/D-SHIFT/NT(X)/U-REDUCE/REDUCE/D-REDUCE/FINISH.",
        "",
        "JSONL files contain one object per sentence: {sent_id, actions}. TXT files contain one whitespace-separated action sequence per line.",
        "",
    ]
    (OUT / "README.txt").write_text("\n".join(readme), encoding="utf-8")

    for corpus in ["PTB", "CTB"]:
        for split in ["dev", "test"]:
            for system in ["Binary", "Arity", "Delimiter"]:
                trees = tree_reader.read_treebank(str(FILES[(corpus, split, system)]))
                dump_implementation_oracle(corpus, split, system, trees)
                if system == "Delimiter":
                    dump_delimiter_paper_oracle(corpus, split, trees)


if __name__ == "__main__":
    main()
