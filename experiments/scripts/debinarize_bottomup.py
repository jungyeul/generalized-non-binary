#!/usr/bin/env python3
from stanza.models.constituency import tree_reader
from stanza.models.constituency.parse_tree import Tree


def expand_plus(label, children):
    parts = label.split("+")
    node_children = children
    for part in reversed(parts):
        node = Tree(part, node_children)
        node_children = [node]
    return node_children[0]


def debinarize(node):
    if node.is_leaf():
        return Tree(node.label)

    children = []
    for child in node.children:
        new_child = debinarize(child)
        if new_child.label.startswith("@") and not new_child.is_leaf():
            children.extend(new_child.children)
        else:
            children.append(new_child)

    if node.label.startswith("@"):
        return Tree(node.label, children)
    if "+" in node.label:
        return expand_plus(node.label, children)
    return Tree(node.label, children)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    trees = tree_reader.read_treebank(args.input)
    with open(args.output, "w", encoding="utf-8") as fout:
        for tree in trees:
            fout.write("{:_O}\n".format(debinarize(tree)))


if __name__ == "__main__":
    main()
