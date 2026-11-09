"""Canonical non-binary IO from the user's canonical_IO.ipynb.

SHIFT pushes a bare word. NT(label) follows the completed left corner,
including a bare terminal for POS. REDUCE combines that left corner and the
completed suffix above the marker, without a label or arity in the action.
FINISH is explicit. This system is for future reference, outside the BU/TD
scope of the dynamic-oracle paper. Native Stanza IO actions/oracles are unused.
"""
from stanza.models.constituency.parse_transitions import (
    Transition, TransitionScheme, Dummy, CloseConstituent, too_many_unary_nodes,
)


def enabled(model):
    return model.transition_scheme is TransitionScheme.CANONICAL_IN_ORDER


def root_complete(state, model):
    if state.num_opens or not state.empty_word_queue() or not state.has_one_constituent():
        return False
    tree = model.get_top_constituent(state.constituents)
    return not isinstance(tree, Dummy) and not tree.is_leaf() and tree.label in model.root_labels


def has_bare_terminal(state, model):
    stack = state.constituents
    while stack.parent is not None:
        tree = model.get_top_constituent(stack)
        if not isinstance(tree, Dummy) and tree.is_leaf():
            return True
        stack = stack.pop()
    return False


class CanonicalIOShift(Transition):
    def is_legal(self, state, model):
        if (not enabled(model) or state.finished(model) or state.empty_word_queue()
                or root_complete(state, model) or has_bare_terminal(state, model)):
            return False
        # With no open parent, a second completed sibling cannot be attached
        # to the first: introduce its parent before shifting that sibling.
        return state.empty_constituents or state.num_opens > 0

    def update_state(self, state, model):
        return (state.word_position + 1, state.constituents,
                model.transform_word_to_constituent(state), None)

    def short_name(self):
        return 'SHIFT'

    def __repr__(self):
        return 'CanonicalIOShift'

    def __eq__(self, other):
        return isinstance(other, CanonicalIOShift)

    def __hash__(self):
        return hash('CanonicalIOShift')


class CanonicalIONT(Transition):
    def __init__(self, label):
        if not isinstance(label, str) or not label:
            raise ValueError('CanonicalIONT requires a nonempty label')
        self.label = (label,)

    def delta_opens(self):
        return 1

    def is_legal(self, state, model):
        if not enabled(model) or state.finished(model) or state.empty_constituents or root_complete(state, model):
            return False
        left = model.get_top_constituent(state.constituents)
        if isinstance(left, Dummy):
            return False  # NT requires a left-corner tree, not another marker
        if self.label[0] in model.root_labels:
            # The outer root may begin after its first completed child, even
            # before all input has been read (non-unary roots are supported).
            return state.num_opens == 0 and state.has_one_constituent() and not left.is_leaf()
        if state.num_opens >= (state.sentence_length + 1) * (model.unary_limit + 2):
            return False
        # Do not open an impossible unary once there are no future siblings.
        if state.empty_word_queue() and not left.is_leaf() and too_many_unary_nodes(left, model.unary_limit):
            return False
        return True

    def update_state(self, state, model):
        return state.word_position, state.constituents, Dummy(self.label), CanonicalIONT

    @staticmethod
    def build_constituents(model, data):
        return model.dummy_constituents(data)

    def short_name(self):
        return 'NT'

    def __repr__(self):
        return 'CanonicalIONT(%r)' % self.label[0]

    def __eq__(self, other):
        return isinstance(other, CanonicalIONT) and self.label == other.label

    def __hash__(self):
        return hash(('CanonicalIONT', self.label))


class CanonicalIOReduce(Transition):
    def delta_opens(self):
        return -1

    def is_legal(self, state, model):
        if not enabled(model) or state.finished(model) or state.num_opens <= 0:
            return False
        stack = state.constituents
        right = []
        while stack.parent is not None:
            tree = model.get_top_constituent(stack)
            if isinstance(tree, Dummy):
                marker = tree
                break
            right.append(tree)
            stack = stack.pop()
        else:
            return False
        stack = stack.pop()
        if stack.parent is None:
            return False
        left = model.get_top_constituent(stack)
        if isinstance(left, Dummy):
            return False
        if left.is_leaf():
            return not right and marker.label[0] not in model.root_labels
        if any(child.is_leaf() or isinstance(child, Dummy) for child in right):
            return False
        if marker.label[0] in model.root_labels:
            return state.empty_word_queue() and state.num_opens == 1 and stack.length == 2
        if not right and too_many_unary_nodes(left, model.unary_limit):
            return False
        return True

    def update_state(self, state, model):
        children = []
        stack = state.constituents
        while not isinstance(model.get_top_constituent(stack), Dummy):
            children.append(stack.value)
            stack = stack.pop()
        label = model.get_top_constituent(stack).label
        stack = stack.pop()  # marker
        children.append(stack.value)  # left corner belongs to this parent
        stack = stack.pop()
        children.reverse()
        return state.word_position, stack, (label, children), CloseConstituent

    def short_name(self):
        return 'REDUCE'

    def __repr__(self):
        return 'CanonicalIOReduce'

    def __eq__(self, other):
        return isinstance(other, CanonicalIOReduce)

    def __hash__(self):
        return hash('CanonicalIOReduce')


class CanonicalIOFinish(Transition):
    def is_legal(self, state, model):
        return enabled(model) and not state.finished(model) and root_complete(state, model)

    def update_state(self, state, model):
        return state.word_position, state.constituents, None, None

    def short_name(self):
        return 'FINISH'

    def __repr__(self):
        return 'CanonicalIOFinish'

    def __eq__(self, other):
        return isinstance(other, CanonicalIOFinish)

    def __hash__(self):
        return hash('CanonicalIOFinish')


def yield_canonical_in_order_sequence(tree):
    if tree.is_leaf() or not tree.children:
        raise ValueError('Canonical IO requires a nonempty parent for each word')
    if tree.is_preterminal():
        yield CanonicalIOShift()
    else:
        yield from yield_canonical_in_order_sequence(tree.children[0])
    yield CanonicalIONT(tree.label)
    for child in tree.children[1:]:
        yield from yield_canonical_in_order_sequence(child)
    yield CanonicalIOReduce()
