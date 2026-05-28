#!/usr/bin/env python3
from collections import Counter
from pathlib import Path

from stanza.models.constituency import tree_reader


ROOT = Path("/home/zihopc/vscode/Jungyeul/20251116_stanza")
ROOT_LABELS = {"ROOT", "TOP"}
BUCKETS = ["k=1", "k=2", "k=3", "k=4", "k>=5"]

DATA = {
    "PTB": {
        "gold": ROOT / "base_dir/data/constituency/en_ptb_test.mrg",
        "A": ROOT / "bottomup_runs/predictions/arity_test.validyield_fallback.pred.mrg",
        "D": ROOT / "bottomup_runs/predictions/delimiter_test.pred.mrg",
    },
    "CTB": {
        "gold": ROOT / "data_ctb/CTB_5.1/zh_ctb_test.mrg",
        "A": ROOT / "bottomup_runs_ctb/predictions/arity_test.pred.mrg",
        "D": ROOT / "bottomup_runs_ctb/predictions/delimiter_test.pred.mrg",
    },
}


def bucket(arity):
    return f"k={arity}" if arity <= 4 else "k>=5"


def read(path):
    return tree_reader.read_treebank(str(path))


def constituents_by_arity(tree, sent_id):
    counts = {key: Counter() for key in BUCKETS}

    def walk(node, start, is_root=False):
        if node.is_leaf():
            return start + 1
        if node.is_preterminal():
            return start + len(node.leaf_labels())

        cur = start
        for child in node.children:
            cur = walk(child, cur, False)
        end = cur

        if not (is_root and node.label in ROOT_LABELS):
            counts[bucket(len(node.children))][(sent_id, start, end, node.label)] += 1
        return end

    walk(tree, 0, True)
    return counts


def corpus_constituents_by_arity(trees):
    total = {key: Counter() for key in BUCKETS}
    for sent_id, tree in enumerate(trees):
        per_tree = constituents_by_arity(tree, sent_id)
        for key in BUCKETS:
            total[key].update(per_tree[key])
    return total


def score(gold_counts, pred_counts):
    matched = sum((gold_counts & pred_counts).values())
    pred_total = sum(pred_counts.values())
    gold_total = sum(gold_counts.values())
    precision = matched / pred_total * 100 if pred_total else 0.0
    recall = matched / gold_total * 100 if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return matched, pred_total, gold_total, precision, recall, f1


def main():
    print("corpus\tsystem\tbucket\tmatched\tpred\tgold\tprecision\trecall\tf1")
    summary = {}
    for corpus, paths in DATA.items():
        gold_trees = read(paths["gold"])
        gold = corpus_constituents_by_arity(gold_trees)
        for system in ["A", "D"]:
            pred_trees = read(paths[system])
            if len(pred_trees) != len(gold_trees):
                raise ValueError(f"{corpus} {system}: gold/pred length mismatch")
            for idx, (gold_tree, pred_tree) in enumerate(zip(gold_trees, pred_trees)):
                if gold_tree.leaf_labels() != pred_tree.leaf_labels():
                    raise ValueError(f"{corpus} {system}: leaf mismatch at sentence {idx}")
            pred = corpus_constituents_by_arity(pred_trees)
            summary[(corpus, system)] = {}
            for key in BUCKETS:
                matched, pred_total, gold_total, precision, recall, f1 = score(gold[key], pred[key])
                summary[(corpus, system)][key] = f1
                print(
                    f"{corpus}\t{system}\t{key}\t{matched}\t{pred_total}\t{gold_total}\t"
                    f"{precision:.4f}\t{recall:.4f}\t{f1:.4f}"
                )
    print()
    print("latex rows")
    for corpus in ["PTB", "CTB"]:
        for system in ["A", "D"]:
            values = " & ".join(f"{summary[(corpus, system)][key]:.2f}" for key in BUCKETS)
            print(f"{corpus} {system}: {values}")


if __name__ == "__main__":
    main()
