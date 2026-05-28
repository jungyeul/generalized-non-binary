#!/usr/bin/env python3
from collections import Counter
from pathlib import Path

from stanza.models.constituency import tree_reader


ROOT = Path("/home/zihopc/vscode/Jungyeul/20251116_stanza")

SETS = {
    "ptb_dev": ROOT / "base_dir/data/constituency/en_ptb_dev.mrg",
    "ptb_test": ROOT / "base_dir/data/constituency/en_ptb_test.mrg",
    "ctb_dev": ROOT / "data_ctb/CTB_5.1/zh_ctb_dev.mrg",
    "ctb_test": ROOT / "data_ctb/CTB_5.1/zh_ctb_test.mrg",
}

PRED = {
    "ptb_binary_dev_raw": ROOT / "bottomup_runs/predictions/binary_dev.pred.mrg",
    "ptb_binary_dev_debin": ROOT / "bottomup_runs/predictions/binary_dev.debin.pred.mrg",
    "ptb_binary_test_raw": ROOT / "bottomup_runs/predictions/binary_test.pred.mrg",
    "ptb_binary_test_debin": ROOT / "bottomup_runs/predictions/binary_test.debin.pred.mrg",
    "ctb_binary_dev_raw": ROOT / "bottomup_runs_ctb/predictions/binary_dev.pred.mrg",
    "ctb_binary_dev_debin": ROOT / "bottomup_runs_ctb/predictions/binary_dev.debin.pred.mrg",
    "ctb_binary_test_raw": ROOT / "bottomup_runs_ctb/predictions/binary_test.pred.mrg",
    "ctb_binary_test_debin": ROOT / "bottomup_runs_ctb/predictions/binary_test.debin.pred.mrg",
    "ptb_arity_test_raw": ROOT / "bottomup_runs/predictions/arity_test.pred.mrg",
    "ptb_arity_test_fallback": ROOT / "bottomup_runs/predictions/arity_test.validyield_fallback.pred.mrg",
}


def read(path):
    return tree_reader.read_treebank(str(path))


def read_allow_illegal_root(path):
    return [tree.prune_none().simplify_labels() for tree in tree_reader.read_tree_file(str(path))]


def walk_labels(tree, labels):
    if tree.is_leaf():
        return
    labels.append(tree.label)
    for child in tree.children:
        walk_labels(child, labels)


def label_stats(trees):
    labels = []
    for tree in trees:
        walk_labels(tree, labels)
    return Counter(labels)


def root_issue(tree):
    if tree.is_leaf():
        return "leaf-root"
    if tree.label not in {"ROOT", "TOP"}:
        return f"root-label={tree.label}"
    if len(tree.children) != 1:
        return f"root-arity={len(tree.children)}"
    return None


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        return min(len(a), len(b)), "<END>" if len(a) <= len(b) else a[min(len(a), len(b))], "<END>" if len(b) <= len(a) else b[min(len(a), len(b))]
    return None


def compare_yields(name, gold_trees, pred_trees, max_examples=8):
    leaf_mismatch = []
    root_issues = []
    pred_len = Counter()
    gold_len = Counter()
    for idx, (gold, pred) in enumerate(zip(gold_trees, pred_trees)):
        gold_leaves = gold.leaf_labels()
        pred_leaves = pred.leaf_labels()
        gold_len[len(gold_leaves)] += 1
        pred_len[len(pred_leaves)] += 1
        if gold_leaves != pred_leaves:
            leaf_mismatch.append((idx, len(gold_leaves), len(pred_leaves), first_diff(gold_leaves, pred_leaves), gold_leaves[:20], pred_leaves[:20]))
        issue = root_issue(pred)
        if issue:
            root_issues.append((idx, issue, pred.label, len(pred.children) if not pred.is_leaf() else 0))

    labels = label_stats(pred_trees)
    temp_labels = {k: v for k, v in labels.items() if str(k).startswith("@") or "+" in str(k)}
    print(f"\n## {name}")
    print(f"gold_n={len(gold_trees)} pred_n={len(pred_trees)}")
    print(f"leaf_mismatches={len(leaf_mismatch)}")
    print(f"root_issues={len(root_issues)}")
    print(f"temporary_or_compound_labels={len(temp_labels)} total_occurrences={sum(temp_labels.values())}")
    if temp_labels:
        print("top_temp_labels", Counter(temp_labels).most_common(12))
    if leaf_mismatch:
        print("leaf_mismatch_examples")
        for item in leaf_mismatch[:max_examples]:
            idx, glen, plen, diff, ghead, phead = item
            print(f"  sent={idx} gold_len={glen} pred_len={plen} first_diff={diff}")
            print(f"    gold_head={ghead}")
            print(f"    pred_head={phead}")
    if root_issues:
        print("root_issue_examples", root_issues[:max_examples])


def arity_failed_report():
    gold = read(SETS["ptb_test"])
    raw = read_allow_illegal_root(PRED["ptb_arity_test_raw"])
    fallback = read(PRED["ptb_arity_test_fallback"])
    print("\n## PTB arity raw invalid/fallback report")
    failed = []
    for idx, (g, p, fb) in enumerate(zip(gold, raw, fallback)):
        issues = []
        if g.leaf_labels() != p.leaf_labels():
            issues.append("yield-mismatch")
        issue = root_issue(p)
        if issue:
            issues.append(issue)
        if issues:
            failed.append((idx, issues, len(g.leaf_labels()), len(p.leaf_labels()), first_diff(g.leaf_labels(), p.leaf_labels()), p.label, len(p.children) if not p.is_leaf() else 0))
    print(f"failed={len(failed)}")
    print("issue_counts", Counter(issue for _, issues, *_ in failed for issue in issues))
    for item in failed[:30]:
        idx, issues, glen, plen, diff, label, arity = item
        print(f"  sent={idx} issues={issues} gold_len={glen} pred_len={plen} first_diff={diff} root={label}/{arity}")
        print(f"    gold={' '.join(gold[idx].leaf_labels()[:40])}")
        print(f"    pred={' '.join(raw[idx].leaf_labels()[:40])}")
        print(f"    fallback_root={fallback[idx].label}/{len(fallback[idx].children) if not fallback[idx].is_leaf() else 0} fallback_len={len(fallback[idx].leaf_labels())}")


def main():
    datasets = {
        "PTB binary dev raw": (read(SETS["ptb_dev"]), read(PRED["ptb_binary_dev_raw"])),
        "PTB binary dev debin": (read(SETS["ptb_dev"]), read(PRED["ptb_binary_dev_debin"])),
        "PTB binary test raw": (read(SETS["ptb_test"]), read(PRED["ptb_binary_test_raw"])),
        "PTB binary test debin": (read(SETS["ptb_test"]), read(PRED["ptb_binary_test_debin"])),
        "CTB binary dev raw": (read(SETS["ctb_dev"]), read(PRED["ctb_binary_dev_raw"])),
        "CTB binary dev debin": (read(SETS["ctb_dev"]), read(PRED["ctb_binary_dev_debin"])),
        "CTB binary test raw": (read(SETS["ctb_test"]), read(PRED["ctb_binary_test_raw"])),
        "CTB binary test debin": (read(SETS["ctb_test"]), read(PRED["ctb_binary_test_debin"])),
    }
    for name, (gold, pred) in datasets.items():
        compare_yields(name, gold, pred)
    arity_failed_report()


if __name__ == "__main__":
    main()
