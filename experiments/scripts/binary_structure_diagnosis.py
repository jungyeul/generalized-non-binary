#!/usr/bin/env python3
from collections import Counter
from pathlib import Path

from stanza.models.constituency import tree_reader


ROOT = Path("/home/zihopc/vscode/Jungyeul/20251116_stanza")
ROOT_LABELS = {"ROOT", "TOP"}

PAIRS = {
    "PTB binary dev debin": (
        ROOT / "base_dir/data/constituency/en_ptb_dev.mrg",
        ROOT / "bottomup_runs/predictions/binary_dev.debin.pred.mrg",
    ),
    "PTB binary test debin": (
        ROOT / "base_dir/data/constituency/en_ptb_test.mrg",
        ROOT / "bottomup_runs/predictions/binary_test.debin.pred.mrg",
    ),
    "CTB binary dev debin": (
        ROOT / "data_ctb/CTB_5.1/zh_ctb_dev.mrg",
        ROOT / "bottomup_runs_ctb/predictions/binary_dev.debin.pred.mrg",
    ),
    "CTB binary test debin": (
        ROOT / "data_ctb/CTB_5.1/zh_ctb_test.mrg",
        ROOT / "bottomup_runs_ctb/predictions/binary_test.debin.pred.mrg",
    ),
    "CTB arity test": (
        ROOT / "data_ctb/CTB_5.1/zh_ctb_test.mrg",
        ROOT / "bottomup_runs_ctb/predictions/arity_test.pred.mrg",
    ),
    "CTB delimiter test": (
        ROOT / "data_ctb/CTB_5.1/zh_ctb_test.mrg",
        ROOT / "bottomup_runs_ctb/predictions/delimiter_test.pred.mrg",
    ),
}


def read(path):
    return tree_reader.read_treebank(str(path))


def spans(tree):
    out = Counter()

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
            out[(start, end, node.label)] += 1
        return end

    walk(tree, 0, True)
    return out


def score(g, p):
    m = sum((g & p).values())
    pt = sum(p.values())
    gt = sum(g.values())
    prec = m / pt * 100 if pt else 0.0
    rec = m / gt * 100 if gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return m, pt, gt, prec, rec, f1


def sent_bucket(n):
    if n <= 10:
        return "<=10"
    if n <= 20:
        return "11-20"
    if n <= 40:
        return "21-40"
    return ">40"


def main():
    for name, (gold_path, pred_path) in PAIRS.items():
        gold = read(gold_path)
        pred = read(pred_path)
        total_g = Counter()
        total_p = Counter()
        by_bucket = {}
        exact = 0
        leaf_mismatch = 0
        worst = []
        for i, (g, p) in enumerate(zip(gold, pred)):
            if g.leaf_labels() != p.leaf_labels():
                leaf_mismatch += 1
                continue
            gs = spans(g)
            ps = spans(p)
            if gs == ps:
                exact += 1
            total_g.update({(i,) + key: value for key, value in gs.items()})
            total_p.update({(i,) + key: value for key, value in ps.items()})
            b = sent_bucket(len(g.leaf_labels()))
            by_bucket.setdefault(b, [Counter(), Counter(), 0])
            by_bucket[b][0].update({(i,) + key: value for key, value in gs.items()})
            by_bucket[b][1].update({(i,) + key: value for key, value in ps.items()})
            by_bucket[b][2] += 1
            worst.append((score(gs, ps)[5], i, len(g.leaf_labels()), sum(gs.values()), sum(ps.values())))

        m, pt, gt, prec, rec, f1 = score(total_g, total_p)
        print(f"\n## {name}")
        print(f"sentences={len(gold)} leaf_mismatch_skipped={leaf_mismatch} exact={exact}")
        print(f"matched={m} pred={pt} gold={gt} precision={prec:.2f} recall={rec:.2f} f1={f1:.2f}")
        for b in ["<=10", "11-20", "21-40", ">40"]:
            if b in by_bucket:
                bg, bp, n = by_bucket[b]
                bm, bpt, bgt, bprec, brec, bf1 = score(bg, bp)
                print(f"  {b}: n={n} f1={bf1:.2f} P={bprec:.2f} R={brec:.2f} pred/gold={bpt}/{bgt}")
        print("  worst_valid_sentences", sorted(worst)[:8])


if __name__ == "__main__":
    main()
