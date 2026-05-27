"""
Build a transition sequence from parse trees.

Supports multiple transition schemes - TOP_DOWN and variants, IN_ORDER
"""

import logging

from stanza.models.common import utils
from stanza.models.constituency.parse_transitions import Shift, CompoundUnary, OpenConstituent, CloseConstituent, TransitionScheme, Finalize
from stanza.models.constituency.parse_transitions import BottomUpUnary, BottomUpBinary, BottomUpArity, BottomUpDShift, BottomUpReduce, BottomUpDReduce
from stanza.models.constituency.tree_reader import read_trees
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()

logger = logging.getLogger('stanza.constituency.trainer')

def yield_bottom_up_binary_sequence(tree, default_direction="RIGHT"):
    """
    Bottom-up transition sequence for an already binarized tree.
    """
    if tree.is_preterminal():
        yield Shift()
        return

    if tree.is_leaf():
        return

    for child in tree.children:
        for transition in yield_bottom_up_binary_sequence(child, default_direction):
            yield transition

    if len(tree.children) == 1:
        yield BottomUpUnary(tree.label)
    elif len(tree.children) == 2:
        yield BottomUpBinary(tree.label, default_direction)
    else:
        raise ValueError("BOTTOM_UP_BINARY requires a binary tree.  Node %s has %d children: %s" % (tree.label, len(tree.children), tree))

def yield_bottom_up_arity_sequence(tree):
    """
    Bottom-up non-binary sequence with explicit reduction arity.
    """
    if tree.is_preterminal():
        yield Shift()
        return

    if tree.is_leaf():
        return

    for child in tree.children:
        for transition in yield_bottom_up_arity_sequence(child):
            yield transition

    yield BottomUpArity(tree.label, len(tree.children))

class _DBoundary:
    def __init__(self, target_id):
        self.target_id = target_id

class _TreeNavigator:
    def __init__(self, root):
        self.parent_of = {}
        self.child_index_of = {}
        self._register(root)

    def _register(self, node):
        for index, child in enumerate(node.children):
            self.parent_of[id(child)] = node
            self.child_index_of[id(child)] = index
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

def yield_bottom_up_delimiter_sequence(tree):
    """
    Delimiter-guided non-binary bottom-up sequence.

    The notebook's D-SHIFT(word) is represented as two Stanza transitions:
    BottomUpDShift() followed by the regular Shift(), because Stanza's bulk
    transition API pushes one constituent per transition.
    """
    navigator = _TreeNavigator(tree)
    previous_parent_id = None
    terminal_index = 0

    def shift_word(parent, preterminal):
        nonlocal previous_parent_id, terminal_index

        current_parent_id = id(parent) if parent is not None else None
        is_first_terminal = terminal_index == 0
        parent_changed = previous_parent_id is not None and current_parent_id != previous_parent_id

        target = None
        if preterminal is not None:
            target = navigator.highest_left_edge_phrasal_ancestor(preterminal)

        use_d_shift = target is not None and (is_first_terminal or parent_changed)
        boundary = None
        if use_d_shift:
            yield BottomUpDShift()
            boundary = _DBoundary(id(target))

        yield Shift()
        previous_parent_id = current_parent_id
        terminal_index += 1
        return boundary

    def visit(node, parent=None):
        if node.is_leaf():
            yield from shift_word(parent, preterminal=None)
            return None

        if node.is_preterminal():
            boundary = yield from shift_word(parent, preterminal=node)
            return boundary

        child_boundaries = []
        for child in node.children:
            boundary = yield from visit(child, parent=node)
            child_boundaries.append(boundary)

        active_boundaries = [boundary for boundary in child_boundaries if boundary is not None]
        if not active_boundaries:
            yield BottomUpReduce(node.label)
            return None

        if len(active_boundaries) > 1:
            raise ValueError("Multiple active delimiters reached node %s" % node.label)

        active = active_boundaries[0]
        if active.target_id == id(node):
            yield BottomUpReduce(node.label)
            return None

        yield BottomUpDReduce(node.label)
        return active

    final_boundary = yield from visit(tree)
    if final_boundary is not None:
        raise ValueError("Unconsumed delimiter boundary")

def yield_top_down_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN_UNARY):
    """
    For tree (X A B C D), yield Open(X) A B C D Close

    The details are in how to treat unary transitions
    Three possibilities handled by this method:
      TOP_DOWN_UNARY:    (Y (X ...)) -> Open(X) ... Close Unary(Y)
      TOP_DOWN_COMPOUND: (Y (X ...)) -> Open(Y, X) ... Close
      TOP_DOWN:          (Y (X ...)) -> Open(Y) Open(X) ... Close Close
    """
    if tree.is_preterminal():
        yield Shift()
        return

    if tree.is_leaf():
        return

    if transition_scheme is TransitionScheme.TOP_DOWN_UNARY:
        if len(tree.children) == 1:
            labels = []
            while not tree.is_preterminal() and len(tree.children) == 1:
                labels.append(tree.label)
                tree = tree.children[0]
            for transition in yield_top_down_sequence(tree, transition_scheme):
                yield transition
            yield CompoundUnary(*labels)
            return

    if transition_scheme is TransitionScheme.TOP_DOWN_COMPOUND:
        labels = [tree.label]
        while len(tree.children) == 1 and not tree.children[0].is_preterminal():
            tree = tree.children[0]
            labels.append(tree.label)
        yield OpenConstituent(*labels)
    else:
        yield OpenConstituent(tree.label)
    for child in tree.children:
        for transition in yield_top_down_sequence(child, transition_scheme):
            yield transition
    yield CloseConstituent()

def yield_in_order_sequence(tree):
    """
    For tree (X A B C D), yield A Open(X) B C D Close
    """
    if tree.is_preterminal():
        yield Shift()
        return

    if tree.is_leaf():
        return

    for transition in yield_in_order_sequence(tree.children[0]):
        yield transition

    yield OpenConstituent(tree.label)

    for child in tree.children[1:]:
        for transition in yield_in_order_sequence(child):
            yield transition

    yield CloseConstituent()



def yield_in_order_compound_sequence(tree, transition_scheme):
    def helper(tree):
        if tree.is_leaf():
            return

        labels = []
        while len(tree.children) == 1 and not tree.is_preterminal():
            labels.append(tree.label)
            tree = tree.children[0]

        if tree.is_preterminal():
            yield Shift()
            if len(labels) > 0:
                yield CompoundUnary(*labels)
            return

        for transition in helper(tree.children[0]):
            yield transition

        if transition_scheme is TransitionScheme.IN_ORDER_UNARY:
            yield OpenConstituent(tree.label)
        else:
            labels.append(tree.label)
            yield OpenConstituent(*labels)

        for child in tree.children[1:]:
            for transition in helper(child):
                yield transition

        yield CloseConstituent()

        if transition_scheme is TransitionScheme.IN_ORDER_UNARY and len(labels) > 0:
            yield CompoundUnary(*labels)

    if len(tree.children) == 0:
        raise ValueError("Cannot build {} on an empty tree".format(transition_scheme))
    if len(tree.children) != 1:
        raise ValueError("Cannot build {} with a tree that has two top level nodes: {}".format(transition_scheme, tree))

    for t in helper(tree.children[0]):
        yield t

    yield Finalize(tree.label)

def build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN_UNARY):
    """
    Turn a single tree into a list of transitions based on the TransitionScheme
    """
    if transition_scheme is TransitionScheme.BOTTOM_UP_BINARY:
        return list(yield_bottom_up_binary_sequence(tree))
    elif transition_scheme is TransitionScheme.BOTTOM_UP_ARITY:
        return list(yield_bottom_up_arity_sequence(tree))
    elif transition_scheme is TransitionScheme.BOTTOM_UP_DELIMITER:
        return list(yield_bottom_up_delimiter_sequence(tree))
    elif transition_scheme is TransitionScheme.IN_ORDER:
        return list(yield_in_order_sequence(tree))
    elif (transition_scheme is TransitionScheme.IN_ORDER_COMPOUND or
          transition_scheme is TransitionScheme.IN_ORDER_UNARY):
        return list(yield_in_order_compound_sequence(tree, transition_scheme))
    else:
        return list(yield_top_down_sequence(tree, transition_scheme))

def build_treebank(trees, transition_scheme=TransitionScheme.TOP_DOWN_UNARY, reverse=False):
    """
    Turn each of the trees in the treebank into a list of transitions based on the TransitionScheme
    """
    if reverse:
        return [build_sequence(tree.reverse(), transition_scheme) for tree in trees]
    else:
        return [build_sequence(tree, transition_scheme) for tree in trees]

def all_transitions(transition_lists):
    """
    Given a list of transition lists, combine them all into a list of unique transitions.
    """
    transitions = set()
    for trans_list in transition_lists:
        transitions.update(trans_list)
    return sorted(transitions)

def convert_trees_to_sequences(trees, treebank_name, transition_scheme, reverse=False):
    """
    Wrap both build_treebank and all_transitions, possibly with a tqdm

    Converts trees to a list of sequences, then returns the list of known transitions
    """
    if len(trees) == 0:
        return [], []
    logger.info("Building %s transition sequences", treebank_name)
    if logger.getEffectiveLevel() <= logging.INFO:
        trees = tqdm(trees)
    sequences = build_treebank(trees, transition_scheme, reverse)
    transitions = all_transitions(sequences)
    return sequences, transitions

def main():
    """
    Convert a sample tree and print its transitions
    """
    text="( (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"
    #text = "(WP Who)"

    tree = read_trees(text)[0]

    print(tree)
    transitions = build_sequence(tree)
    print(transitions)

if __name__ == '__main__':
    main()
