#!/usr/bin/env python3
from collections import Counter

from stanza.models.constituency import tree_reader

FILES = {
    "ptb_orig_dev": "base_dir/data/constituency/en_ptb_dev.mrg",
    "ptb_bin_dev": "data/dev.txt",
    "ctb_orig_dev": "data_ctb/CTB_5.1/zh_ctb_dev.mrg",
    "ctb_bin_dev": "data_ctb/CTB_binary/dev_zh.txt",
    "ctb_orig_test": "data_ctb/CTB_5.1/zh_ctb_test.mrg",
}


def preterms(trees):
    counts = Counter()

    def walk(tree):
        if tree.is_leaf():
            return
        if tree.is_preterminal():
            counts[tree.label] += 1
        else:
            for child in tree.children:
                walk(child)

    for tree in trees:
        walk(tree)
    return counts


for name, path in FILES.items():
    trees = tree_reader.read_treebank(path)
    counts = preterms(trees)
    print()
    print(name, "nlabels", len(counts), "tokens", sum(counts.values()))
    print(counts.most_common(25))
