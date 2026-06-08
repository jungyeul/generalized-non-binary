#!/usr/bin/env python3
from collections import Counter
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

FILES = {
    ("PTB", "dev", "Binary"): ROOT / "bottomup_runs/predictions/binary_dev.pred.mrg",
    ("PTB", "test", "Binary"): ROOT / "bottomup_runs/predictions/binary_test.pred.mrg",
    ("PTB", "dev", "Arity"): ROOT / "bottomup_runs/predictions/arity_dev.pred.mrg",
    ("PTB", "test", "Arity"): ROOT / "bottomup_runs/predictions/arity_test.pred.mrg",
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
}


def read_trees(path):
    return tree_reader.read_treebank(str(path))


def action_key(action):
    if isinstance(action, Shift):
        return "SHIFT"
    if isinstance(action, BottomUpUnary):
        return f"REDUCE-UNARY-{action.label}"
    if isinstance(action, BottomUpBinary):
        return f"REDUCE-{action.direction}-{action.label}"
    if isinstance(action, BottomUpArity):
        return f"REDUCE-{action.label}#{action.arity}"
    return repr(action)


def action_family(key):
    if key in {"SHIFT", "FINISH"}:
        return key
    if key.startswith("REDUCE-UNARY"):
        return "REDUCE-UNARY"
    if key.startswith("REDUCE-LEFT") or key.startswith("REDUCE-RIGHT"):
        return "REDUCE-BINARY"
    if key.startswith("REDUCE-"):
        return "REDUCE-ARITY"
    return key


def summarize_binary_or_arity(trees, scheme):
    seqs = transition_sequence.build_treebank(trees, scheme)
    keys = [action_key(action) for seq in seqs for action in seq]
    keys += ["FINISH"] * len(seqs)
    return {
        "sentences": len(seqs),
        "avg_len": (sum(len(seq) for seq in seqs) + len(seqs)) / len(seqs),
        "inventory_size": len(set(keys)),
        "operator_freq": Counter(action_family(key) for key in keys),
    }


def summarize_delimiter_paper_oracle(trees):
    labels = set()
    inv = set()
    freq = Counter()

    for tree in trees:
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

        previous_parent_id = None
        terminal_index = 0

        def add_nt(label):
            labels.add(label)
            inv.add(f"NT({label})")
            freq["NT(X)"] += 1

        def shift_word(parent, preterminal):
            nonlocal previous_parent_id, terminal_index
            current_parent_id = id(parent) if parent is not None else None
            is_first_terminal = terminal_index == 0
            parent_changed = previous_parent_id is not None and current_parent_id != previous_parent_id
            target = highest_left_edge_phrasal_ancestor(preterminal) if preterminal is not None else None
            use_d_shift = target is not None and (is_first_terminal or parent_changed)
            if use_d_shift:
                inv.add("D-SHIFT")
                freq["D-SHIFT"] += 1
                boundary = id(target)
            else:
                inv.add("SHIFT")
                freq["SHIFT"] += 1
                boundary = None
            if preterminal is not None:
                add_nt(preterminal.label)
                inv.add("U-REDUCE")
                freq["U-REDUCE"] += 1
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

            add_nt(node.label)
            active_boundaries = [boundary for boundary in child_boundaries if boundary is not None]
            if active_boundaries and not (len(active_boundaries) == 1 and active_boundaries[0] == id(node)):
                inv.add("D-REDUCE")
                freq["D-REDUCE"] += 1
                boundary = active_boundaries[0]
            else:
                inv.add("REDUCE")
                freq["REDUCE"] += 1
                boundary = None
            return boundary, child_spans[0][0], child_spans[-1][1]

        visit(tree)
        inv.add("FINISH")
        freq["FINISH"] += 1

    return {
        "sentences": len(trees),
        "avg_len": sum(freq.values()) / len(trees),
        "inventory_size": len(inv),
        "nt_labels": len(labels),
        "operator_freq": freq,
    }


def main():
    print("corpus\tsplit\tsystem\tsentences\tavg_oracle_len\tinventory_size\toperator_freq")
    for corpus in ["PTB", "CTB"]:
        for split in ["dev", "test"]:
            for system in ["Binary", "Arity", "Delimiter"]:
                trees = read_trees(FILES[(corpus, split, system)])
                if system in SCHEMES:
                    summary = summarize_binary_or_arity(trees, SCHEMES[system])
                else:
                    summary = summarize_delimiter_paper_oracle(trees)
                print(
                    f"{corpus}\t{split}\t{system}\t{summary['sentences']}\t"
                    f"{summary['avg_len']:.2f}\t{summary['inventory_size']}\t"
                    f"{dict(summary['operator_freq'])}"
                )


if __name__ == "__main__":
    main()
