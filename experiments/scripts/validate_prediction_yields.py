#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict

from stanza.models.constituency import tree_reader


def first_difference(gold, predicted):
    for index, (gold_word, predicted_word) in enumerate(zip(gold, predicted)):
        if gold_word != predicted_word:
            return index, gold_word, predicted_word
    index = min(len(gold), len(predicted))
    gold_word = gold[index] if index < len(gold) else "<END>"
    predicted_word = predicted[index] if index < len(predicted) else "<END>"
    return index, gold_word, predicted_word


def main():
    parser = argparse.ArgumentParser(
        description="Fail when prediction trees do not preserve gold sentence yields."
    )
    parser.add_argument("gold")
    parser.add_argument("prediction")
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    gold_trees = tree_reader.read_treebank(args.gold)
    predicted_trees = [
        tree.prune_none().simplify_labels()
        for tree in tree_reader.read_tree_file(args.prediction)
    ]

    if len(gold_trees) != len(predicted_trees):
        raise SystemExit(
            f"Tree count mismatch: gold={len(gold_trees)} predicted={len(predicted_trees)}"
        )

    gold_yield_to_sentences = defaultdict(list)
    for index, tree in enumerate(gold_trees):
        gold_yield_to_sentences[tuple(tree.leaf_labels())].append(index + 1)

    mismatches = []
    reused_gold_sentences = Counter()
    for index, (gold_tree, predicted_tree) in enumerate(
        zip(gold_trees, predicted_trees), start=1
    ):
        gold_yield = gold_tree.leaf_labels()
        predicted_yield = predicted_tree.leaf_labels()
        if gold_yield == predicted_yield:
            continue
        sources = gold_yield_to_sentences.get(tuple(predicted_yield), [])
        reused_gold_sentences.update(sources)
        mismatches.append(
            (
                index,
                len(gold_yield),
                len(predicted_yield),
                first_difference(gold_yield, predicted_yield),
                sources,
            )
        )

    print(f"gold_trees={len(gold_trees)}")
    print(f"predicted_trees={len(predicted_trees)}")
    print(f"yield_mismatches={len(mismatches)}")
    if reused_gold_sentences:
        print(f"reused_gold_yields={reused_gold_sentences.most_common()}")
    for sentence, gold_len, predicted_len, difference, sources in mismatches[
        : args.max_examples
    ]:
        print(
            f"sentence={sentence} gold_len={gold_len} predicted_len={predicted_len} "
            f"first_difference={difference} predicted_yield_sources={sources}"
        )

    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
