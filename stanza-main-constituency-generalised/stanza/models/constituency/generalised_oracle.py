"""Static generalised oracles copied from the supplied notebook definitions.

Gold boundary target IDs are used only in oracle derivation, never decoding.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

from stanza.models.constituency.parse_tree import Tree
from stanza.models.constituency.tree_reader import read_trees

OracleOp = Tuple[str, Optional[str]]

def parse_ptb(ptb_string: str) -> Tree:
    """Parse one PTB-style bracketed tree into a Stanza constituency Tree."""
    trees = list(read_trees(ptb_string, use_tqdm=False))
    if not trees:
        raise ValueError(f"No tree found in string: {ptb_string!r}")
    if len(trees) != 1:
        raise ValueError(f"Expected exactly one tree, found {len(trees)}.")
    return trees[0]


def label_of(tree: Tree) -> str:
    """Return the label stored on a Stanza Tree node."""
    return tree.label


def children_of(tree: Tree) -> List[Tree]:
    """Return the children of a Stanza Tree node as a list."""
    return list(tree.children or [])


def make_tree(label: str, children: Optional[Sequence[Tree]] = None) -> Tree:
    """Construct a Stanza Tree node.

    Stanza versions expose the same logical constructor, but this helper keeps
    construction localized in case a local environment uses keyword-only fields.
    """
    children = list(children or [])
    try:
        return Tree(label, children)
    except TypeError:
        return Tree(label=label, children=children)


def is_leaf(tree: Tree) -> bool:
    return len(children_of(tree)) == 0


def is_preterminal(tree: Tree) -> bool:
    children = children_of(tree)
    return len(children) == 1 and is_leaf(children[0])


def terminal_word(preterminal: Tree) -> str:
    """Return the terminal word dominated by a preterminal."""
    if not is_preterminal(preterminal):
        raise ValueError(f"Expected a preterminal, found: {tree_to_ptb(preterminal)}")
    return label_of(children_of(preterminal)[0])


def get_sentence(tree: Tree) -> List[str]:
    """Return the terminal yield of a Stanza Tree."""
    if is_leaf(tree):
        return [label_of(tree)]

    words: List[str] = []
    for child in children_of(tree):
        words.extend(get_sentence(child))
    return words


def tree_to_ptb(tree: Tree) -> str:
    """Return a single-line PTB string from a Stanza Tree."""
    if is_leaf(tree):
        return label_of(tree)
    return f"({label_of(tree)} {' '.join(tree_to_ptb(child) for child in children_of(tree))})"


def same_tree(left: Tree, right: Tree) -> bool:
    """Exact structural comparison through the canonical PTB string form."""
    return tree_to_ptb(left) == tree_to_ptb(right)

@dataclass(frozen=True)
class DBoundary:
    """A delimiter introduced by D-SHIFT and consumed by its target node."""

    target_id: int


class TreeNavigator:
    """Small helper for parent and child-index lookup in a Stanza Tree."""

    def __init__(self, root: Tree):
        self.root = root
        self.parent_of: dict[int, Tree] = {}
        self.child_index_of: dict[int, int] = {}
        self._register(root)

    def _register(self, node: Tree) -> None:
        for index, child in enumerate(children_of(node)):
            self.parent_of[id(child)] = node
            self.child_index_of[id(child)] = index
            self._register(child)

    def highest_left_edge_phrasal_ancestor(self, preterminal: Tree) -> Optional[Tree]:
        """
        Return the highest phrasal ancestor reached by moving upward from a
        preterminal while the current node is the first child of its parent.

        The preterminal itself is not a target, because it is built by U-REDUCE.
        """
        current: Tree = preterminal
        target: Optional[Tree] = None

        while id(current) in self.parent_of:
            parent = self.parent_of[id(current)]
            child_index = self.child_index_of[id(current)]

            if child_index != 0:
                break

            if not is_preterminal(parent):
                target = parent

            current = parent

        return target


def get_oracle_bottomup(tree: Tree) -> List[OracleOp]:
    """
    Return the delimiter-guided non-binary bottom-up oracle for a gold tree.

    The oracle is deterministic for a fixed tree. It does not encode arity in
    phrase-level reduction actions.  The final action is always FINISH.
    """
    oracle: List[OracleOp] = []
    navigator = TreeNavigator(tree)

    previous_parent_id: Optional[int] = None
    terminal_index = 0

    def shift_word(word: str, parent: Optional[Tree], preterminal: Optional[Tree]) -> Optional[DBoundary]:
        nonlocal previous_parent_id, terminal_index

        current_parent_id = id(parent) if parent is not None else None
        is_first_terminal = terminal_index == 0
        parent_changed = previous_parent_id is not None and current_parent_id != previous_parent_id

        target = None
        if preterminal is not None:
            target = navigator.highest_left_edge_phrasal_ancestor(preterminal)

        use_d_shift = target is not None and (is_first_terminal or parent_changed)

        if use_d_shift:
            oracle.append(("D-SHIFT", word))
            boundary = DBoundary(target_id=id(target))
        else:
            oracle.append(("SHIFT", word))
            boundary = None

        previous_parent_id = current_parent_id
        terminal_index += 1
        return boundary

    def visit(node: Tree, parent: Optional[Tree] = None) -> Optional[DBoundary]:
        if is_leaf(node):
            return shift_word(label_of(node), parent, preterminal=None)

        if is_preterminal(node):
            boundary = shift_word(terminal_word(node), parent, preterminal=node)
            oracle.append(("NT", label_of(node)))
            oracle.append(("U-REDUCE", None))
            return boundary

        child_boundaries = [visit(child, parent=node) for child in children_of(node)]

        oracle.append(("NT", label_of(node)))
        active_boundaries = [boundary for boundary in child_boundaries if boundary is not None]

        if not active_boundaries:
            oracle.append(("REDUCE", None))
            return None

        if len(active_boundaries) > 1:
            raise ValueError(
                f"Multiple active delimiters reached node {label_of(node)}: {active_boundaries}"
            )

        active = active_boundaries[0]
        if active.target_id == id(node):
            oracle.append(("REDUCE", None))
            return None

        oracle.append(("D-REDUCE", None))
        return active

    final_boundary = visit(tree)
    if final_boundary is not None:
        raise ValueError(f"Unconsumed delimiter boundary: {final_boundary}")

    oracle.append(("FINISH", None))
    return oracle


def get_oracle_topdown(tree: Tree) -> List[OracleOp]:
    """Return the non-binary top-down oracle for a gold tree.

    The oracle is deterministic for a fixed tree.  It does not encode arity in
    phrase-level reduction actions.  The final action is always FINISH.
    """
    oracle: List[OracleOp] = []

    def visit(node: Tree) -> None:
        if is_leaf(node):
            raise ValueError(
                "Top-down oracle should visit terminals through their preterminal parent. "
                f"Unexpected leaf: {label_of(node)!r}"
            )

        if is_preterminal(node):
            oracle.append(("NT", label_of(node)))
            oracle.append(("SHIFT", terminal_word(node)))
            oracle.append(("U-REDUCE", None))
            return

        oracle.append(("NT", label_of(node)))
        for child in children_of(node):
            visit(child)
        oracle.append(("REDUCE", None))

    visit(tree)
    oracle.append(("FINISH", None))
    return oracle


def get_oracle_inorder(tree: Tree) -> List[OracleOp]:
    """Return the delimiter-guided non-binary in-order oracle for a gold tree.

    The oracle is deterministic for a fixed tree.  It does not encode arity in
    phrase-level reduction actions.  The final action is always FINISH.
    """
    oracle: List[OracleOp] = []
    navigator = TreeNavigator(tree)

    previous_parent_id: Optional[int] = None
    terminal_index = 0

    def shift_word(word: str, parent: Optional[Tree], preterminal: Optional[Tree]) -> Optional[DBoundary]:
        nonlocal previous_parent_id, terminal_index

        current_parent_id = id(parent) if parent is not None else None
        is_first_terminal = terminal_index == 0
        parent_changed = previous_parent_id is not None and current_parent_id != previous_parent_id

        target = None
        if preterminal is not None:
            target = navigator.highest_left_edge_phrasal_ancestor(preterminal)

        use_d_shift = target is not None and (is_first_terminal or parent_changed)

        if use_d_shift:
            oracle.append(("D-SHIFT", word))
            boundary = DBoundary(target_id=id(target))
        else:
            oracle.append(("SHIFT", word))
            boundary = None

        previous_parent_id = current_parent_id
        terminal_index += 1
        return boundary

    def visit(node: Tree, parent: Optional[Tree] = None) -> Optional[DBoundary]:
        if is_leaf(node):
            return shift_word(label_of(node), parent, preterminal=None)

        if is_preterminal(node):
            boundary = shift_word(terminal_word(node), parent, preterminal=node)
            oracle.append(("NT", label_of(node)))
            oracle.append(("U-REDUCE", None))
            return boundary

        children = children_of(node)
        if not children:
            raise ValueError(f"Unexpected empty phrasal node: {label_of(node)!r}")

        active_boundaries: List[DBoundary] = []

        first_boundary = visit(children[0], parent=node)
        if first_boundary is not None:
            active_boundaries.append(first_boundary)

        oracle.append(("NT", label_of(node)))

        for child in children[1:]:
            boundary = visit(child, parent=node)
            if boundary is not None:
                active_boundaries.append(boundary)

        if not active_boundaries:
            oracle.append(("REDUCE", None))
            return None

        if len(active_boundaries) > 1:
            raise ValueError(
                f"Multiple active delimiters reached node {label_of(node)}: {active_boundaries}"
            )

        active = active_boundaries[0]
        if active.target_id == id(node):
            oracle.append(("REDUCE", None))
            return None

        oracle.append(("D-REDUCE", None))
        return active

    final_boundary = visit(tree)
    if final_boundary is not None:
        raise ValueError(f"Unconsumed delimiter boundary: {final_boundary}")

    oracle.append(("FINISH", None))
    return oracle
