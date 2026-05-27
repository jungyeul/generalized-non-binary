#!/usr/bin/env python3
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from stanza.models.constituency import tree_reader, transition_sequence
from stanza.models.constituency.parse_transitions import (
    Shift,
    TransitionScheme,
    BottomUpUnary,
    BottomUpBinary,
    BottomUpArity,
    BottomUpDShift,
    BottomUpReduce,
    BottomUpDReduce,
)


ROOT = Path("/home/zihopc/vscode/Jungyeul/20251116_stanza")
OUT = ROOT / "bottomup_table_stats"

ROOT_LABELS = {"ROOT", "TOP"}
ARITY_BUCKETS = ["k=1", "k=2", "k=3", "k=4", "k>=5"]


DATA = {
    "PTB": {
        "binary_train": ROOT / "data/train.txt",
        "binary_dev": ROOT / "data/dev.txt",
        "gold_train": ROOT / "base_dir/data/constituency/en_ptb_train.mrg",
        "gold_dev": ROOT / "base_dir/data/constituency/en_ptb_dev.mrg",
        "gold_test": ROOT / "base_dir/data/constituency/en_ptb_test.mrg",
        "arity_pred_test": ROOT / "bottomup_runs/predictions/arity_test.validyield_fallback.pred.mrg",
        "delimiter_pred_test": ROOT / "bottomup_runs/predictions/delimiter_test.pred.mrg",
    },
    "CTB": {
        "binary_train": ROOT / "data_ctb/CTB_binary/train_zh.txt",
        "binary_dev": ROOT / "data_ctb/CTB_binary/dev_zh.txt",
        "gold_train": ROOT / "data_ctb/CTB_5.1/zh_ctb_train.mrg",
        "gold_dev": ROOT / "data_ctb/CTB_5.1/zh_ctb_dev.mrg",
        "gold_test": ROOT / "data_ctb/CTB_5.1/zh_ctb_test.mrg",
        "arity_pred_test": ROOT / "bottomup_runs_ctb/predictions/arity_test.pred.mrg",
        "delimiter_pred_test": ROOT / "bottomup_runs_ctb/predictions/delimiter_test.pred.mrg",
    },
}


SCHEMES = {
    "Binary bottom-up": TransitionScheme.BOTTOM_UP_BINARY,
    "Arity-specific bottom-up": TransitionScheme.BOTTOM_UP_ARITY,
    "Delimiter-guided bottom-up": TransitionScheme.BOTTOM_UP_DELIMITER,
}


def read_trees(path):
    return tree_reader.read_treebank(str(path))


def action_kind(action):
    if isinstance(action, Shift):
        return "SHIFT"
    if isinstance(action, BottomUpDShift):
        return "D-SHIFT"
    if isinstance(action, BottomUpUnary):
        return "REDUCE-UNARY"
    if isinstance(action, BottomUpBinary):
        return f"REDUCE-{action.direction}"
    if isinstance(action, BottomUpArity):
        return "REDUCE-ARITY"
    if isinstance(action, BottomUpDReduce):
        return "D-REDUCE"
    if isinstance(action, BottomUpReduce):
        return "REDUCE"
    return type(action).__name__


def full_action_key(action):
    if isinstance(action, Shift):
        return "SHIFT"
    if isinstance(action, BottomUpDShift):
        return "D-SHIFT"
    if isinstance(action, BottomUpUnary):
        return ("REDUCE-UNARY", action.label)
    if isinstance(action, BottomUpBinary):
        return (f"REDUCE-{action.direction}", action.label)
    if isinstance(action, BottomUpArity):
        return ("REDUCE-ARITY", action.label, action.arity)
    if isinstance(action, BottomUpDReduce):
        return ("D-REDUCE", action.label)
    if isinstance(action, BottomUpReduce):
        return ("REDUCE", action.label)
    return repr(action)


def reduction_action_key(action):
    if isinstance(action, Shift) or isinstance(action, BottomUpDShift):
        return None
    return full_action_key(action)


def build_sequences(trees, scheme):
    return transition_sequence.build_treebank(trees, scheme)


def summarize_oracle(trees, scheme):
    seqs = build_sequences(trees, scheme)
    lengths = [len(seq) for seq in seqs]
    all_actions = [action for seq in seqs for action in seq]
    full_inventory = {full_action_key(action) for action in all_actions}
    reduction_inventory = {key for action in all_actions if (key := reduction_action_key(action)) is not None}
    kind_counts = Counter(action_kind(action) for action in all_actions)
    return {
        "sentences": len(seqs),
        "total_actions": sum(lengths),
        "avg_len": sum(lengths) / len(lengths),
        "full_inventory_size": len(full_inventory),
        "reduction_inventory_size": len(reduction_inventory),
        "kind_counts": dict(kind_counts),
    }


def tree_arity_counts(trees, exclude_root=True):
    counts = Counter()

    def bucket(k):
        return f"k={k}" if k <= 4 else "k>=5"

    def walk(node, is_root=False):
        if node.is_leaf() or node.is_preterminal():
            return
        if not (exclude_root and is_root and node.label in ROOT_LABELS):
            counts[bucket(len(node.children))] += 1
        for child in node.children:
            walk(child, False)

    for tree in trees:
        walk(tree, True)
    return {bucket: counts[bucket] for bucket in ARITY_BUCKETS}


class Navigator:
    def __init__(self, root):
        self.parent_of = {}
        self.child_index_of = {}
        self._register(root)

    def _register(self, node):
        for idx, child in enumerate(node.children):
            self.parent_of[id(child)] = node
            self.child_index_of[id(child)] = idx
            self._register(child)

    def highest_left_edge_phrasal_ancestor(self, preterminal):
        current = preterminal
        target = None
        while id(current) in self.parent_of:
            parent = self.parent_of[id(current)]
            child_index = self.child_index_of[id(current)]
            if child_index != 0:
                break
            if not parent.is_preterminal():
                target = parent
            current = parent
        return target


def delimiter_decisions(tree):
    navigator = Navigator(tree)
    previous_parent_id = None
    terminal_index = 0
    dshift_positions = set()
    reductions = {}

    def shift_word(parent, preterminal):
        nonlocal previous_parent_id, terminal_index
        current_parent_id = id(parent) if parent is not None else None
        is_first_terminal = terminal_index == 0
        parent_changed = previous_parent_id is not None and current_parent_id != previous_parent_id
        target = navigator.highest_left_edge_phrasal_ancestor(preterminal) if preterminal is not None else None
        use_d_shift = target is not None and (is_first_terminal or parent_changed)
        boundary = None
        if use_d_shift:
            dshift_positions.add(terminal_index)
            boundary = id(target)
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

        start = child_spans[0][0]
        end = child_spans[-1][1]
        active_boundaries = [boundary for boundary in child_boundaries if boundary is not None]
        if not active_boundaries:
            op = "REDUCE"
            boundary = None
        elif len(active_boundaries) > 1:
            op = "MULTIPLE-ACTIVE"
            boundary = active_boundaries[0]
        elif active_boundaries[0] == id(node):
            op = "REDUCE"
            boundary = None
        else:
            op = "D-REDUCE"
            boundary = active_boundaries[0]

        if not (node.label in ROOT_LABELS and parent is None):
            reductions[(start, end, node.label)] = op
        return boundary, start, end

    visit(tree)
    return dshift_positions, reductions


def delimiter_error_counts(gold_trees, pred_trees):
    counts = Counter()
    matched_reductions = 0
    sentences = 0
    leaf_mismatches = 0
    for gold, pred in zip(gold_trees, pred_trees):
        sentences += 1
        if gold.leaf_labels() != pred.leaf_labels():
            leaf_mismatches += 1
            continue
        gold_dshift, gold_reductions = delimiter_decisions(gold)
        pred_dshift, pred_reductions = delimiter_decisions(pred)

        counts["Missing D-SHIFT"] += len(gold_dshift - pred_dshift)
        counts["Spurious D-SHIFT"] += len(pred_dshift - gold_dshift)

        for key in set(gold_reductions) & set(pred_reductions):
            matched_reductions += 1
            gold_op = gold_reductions[key]
            pred_op = pred_reductions[key]
            if gold_op == "D-REDUCE" and pred_op == "REDUCE":
                counts["Premature REDUCE"] += 1
            elif gold_op == "REDUCE" and pred_op == "D-REDUCE":
                counts["Spurious D-REDUCE"] += 1

    return {
        "sentences": sentences,
        "leaf_mismatches_skipped": leaf_mismatches,
        "matched_reductions": matched_reductions,
        "errors": {k: counts[k] for k in ["Missing D-SHIFT", "Spurious D-SHIFT", "Premature REDUCE", "Spurious D-REDUCE"]},
    }


def format_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))
    out = []
    out.append("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    out.append("| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |")
    return "\n".join(out)


def main():
    OUT.mkdir(exist_ok=True)
    results = {"table2": {}, "table4": {}, "table5": {}, "table6": {}}

    for corpus, paths in DATA.items():
        oracle_inputs = {
            "Binary bottom-up": paths["binary_train"],
            "Arity-specific bottom-up": paths["gold_train"],
            "Delimiter-guided bottom-up": paths["gold_train"],
        }
        for system, scheme in SCHEMES.items():
            trees = read_trees(oracle_inputs[system])
            summary = summarize_oracle(trees, scheme)
            results["table2"].setdefault(system, {})[corpus] = summary["avg_len"]
            results["table4"].setdefault(system, {})[corpus] = {
                "full_inventory_size": summary["full_inventory_size"],
                "reduction_inventory_size": summary["reduction_inventory_size"],
                "kind_counts": summary["kind_counts"],
            }

        gold_test = read_trees(paths["gold_test"])
        arity_pred = read_trees(paths["arity_pred_test"])
        delimiter_pred = read_trees(paths["delimiter_pred_test"])
        results["table5"][corpus] = {
            "G": tree_arity_counts(gold_test),
            "A": tree_arity_counts(arity_pred),
            "D": tree_arity_counts(delimiter_pred),
        }
        results["table6"][corpus] = delimiter_error_counts(gold_test, delimiter_pred)

    (OUT / "tables_2_6.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    lines = []
    lines.append("# Bottom-up Tables 2-6\n")
    lines.append("Table 2 and Table 4 use training gold oracle sequences. Table 5 and Table 6 use test gold/predicted trees. Root/TOP nodes and preterminals are excluded from Table 5 subtree arity counts. PTB arity predictions use the conservative valid-yield fallback file for the 24 invalid ROOT/yield cases found during audit.\n")

    rows = []
    for system in SCHEMES:
        rows.append([system, f"{results['table2'][system]['PTB']:.2f}", f"{results['table2'][system]['CTB']:.2f}"])
    lines.append("## Table 2: Average transition sequence length per sentence")
    lines.append(format_table(["System", "PTB", "CTB"], rows))
    lines.append("")

    rows = []
    for system in SCHEMES:
        ptb = results["table4"][system]["PTB"]
        ctb = results["table4"][system]["CTB"]
        rows.append([system, ptb["full_inventory_size"], ctb["full_inventory_size"], ptb["reduction_inventory_size"], ctb["reduction_inventory_size"]])
    lines.append("## Table 4: Action inventory size used in gold oracle sequences")
    lines.append(format_table(["System", "PTB full", "CTB full", "PTB reductions", "CTB reductions"], rows))
    lines.append("")

    for corpus in ["PTB", "CTB"]:
        rows = []
        for source in ["G", "A", "D"]:
            counts = results["table5"][corpus][source]
            rows.append([source] + [counts[bucket] for bucket in ARITY_BUCKETS])
        lines.append(f"## Table 5: {corpus} subtree arity distribution")
        lines.append(format_table(["Source"] + ARITY_BUCKETS, rows))
        lines.append("")

    rows = []
    for err in ["Missing D-SHIFT", "Spurious D-SHIFT", "Premature REDUCE", "Spurious D-REDUCE"]:
        rows.append([err, results["table6"]["PTB"]["errors"][err], results["table6"]["CTB"]["errors"][err]])
    lines.append("## Table 6: Boundary prediction errors in delimiter-guided parser")
    lines.append(format_table(["Error type", "PTB", "CTB"], rows))
    lines.append("")
    lines.append("Diagnostics:")
    lines.append(f"- PTB Table 6 skipped {results['table6']['PTB']['leaf_mismatches_skipped']} sentence(s) due to delimiter prediction leaf mismatches.")
    lines.append(f"- CTB Table 6 skipped {results['table6']['CTB']['leaf_mismatches_skipped']} sentence(s) due to delimiter prediction leaf mismatches.")

    (OUT / "tables_2_6.md").write_text("\n".join(lines), encoding="utf-8")

    with (OUT / "tables_2_6.tsv").open("w", encoding="utf-8") as fout:
        fout.write("table\tcorpus\tsystem_or_source\tmetric\tvalue\n")
        for system in SCHEMES:
            for corpus in ["PTB", "CTB"]:
                fout.write(f"2\t{corpus}\t{system}\tavg_transition_length\t{results['table2'][system][corpus]:.6f}\n")
        for system in SCHEMES:
            for corpus in ["PTB", "CTB"]:
                item = results["table4"][system][corpus]
                fout.write(f"4\t{corpus}\t{system}\tfull_inventory_size\t{item['full_inventory_size']}\n")
                fout.write(f"4\t{corpus}\t{system}\treduction_inventory_size\t{item['reduction_inventory_size']}\n")
        for corpus in ["PTB", "CTB"]:
            for source in ["G", "A", "D"]:
                for bucket in ARITY_BUCKETS:
                    fout.write(f"5\t{corpus}\t{source}\t{bucket}\t{results['table5'][corpus][source][bucket]}\n")
        for corpus in ["PTB", "CTB"]:
            for err, value in results["table6"][corpus]["errors"].items():
                fout.write(f"6\t{corpus}\tDelimiter-guided\t{err}\t{value}\n")

    print((OUT / "tables_2_6.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
