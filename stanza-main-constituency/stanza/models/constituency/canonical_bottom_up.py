"""Explicit-arity non-binary BU from the user's canonical_BU.ipynb.

SHIFT pushes a bare word; REDUCE(label, arity) builds every node, including
POS and ROOT; FINISH is explicit. Neural word features and composition are
shared with canonical TD. The legacy supplied-POS BOTTOM_UP scheme remains.
"""
from stanza.models.constituency.parse_transitions import (
    Transition, TransitionScheme, CloseConstituent, too_many_unary_nodes,
)


def enabled(model):
    return model.transition_scheme is TransitionScheme.CANONICAL_BOTTOM_UP


def root_complete(state, model):
    if not state.empty_word_queue() or not state.has_one_constituent():
        return False
    tree = model.get_top_constituent(state.constituents)
    return not tree.is_leaf() and tree.label in model.root_labels


class CanonicalBUShift(Transition):
    def is_legal(self, state, model):
        if not enabled(model) or state.finished(model) or state.empty_word_queue():
            return False
        # Require a leaf-parent before shifting another word. Both canonical
        # systems produce complete POS nodes instead of bare children mixed
        # with phrase children. All notebook gold derivations satisfy this.
        return (state.empty_constituents
                or not model.get_top_constituent(state.constituents).is_leaf())

    def update_state(self, state, model):
        return (state.word_position + 1, state.constituents,
                model.transform_word_to_constituent(state), None)

    def short_name(self):
        return 'SHIFT'

    def __repr__(self):
        return 'CanonicalBUShift'

    def __eq__(self, other):
        return isinstance(other, CanonicalBUShift)

    def __hash__(self):
        return hash('CanonicalBUShift')


class CanonicalBUReduce(Transition):
    def __init__(self, label, arity):
        if not isinstance(label, str) or not label:
            raise ValueError('CanonicalBUReduce requires a nonempty label')
        if isinstance(arity, bool) or not isinstance(arity, int) or arity < 1:
            raise ValueError('CanonicalBUReduce requires a positive integer arity')
        self.label = (label,)
        self.arity = arity

    def is_legal(self, state, model):
        if (not enabled(model) or state.finished(model) or root_complete(state, model)
                or state.num_constituents() < self.arity):
            return False
        stack = state.constituents
        children = []
        for _ in range(self.arity):
            children.append(model.get_top_constituent(stack))
            stack = stack.pop()
        if any(child.is_leaf() for child in children):
            # A bare word must become a single-child leaf-parent. The action
            # predicts its label, including POS, just as in the notebook.
            return self.arity == 1 and self.label[0] not in model.root_labels
        if self.label[0] in model.root_labels:
            return state.empty_word_queue() and state.num_constituents() == self.arity
        if self.arity == 1:
            return not too_many_unary_nodes(children[0], model.unary_limit)
        return True

    def update_state(self, state, model):
        children = []
        stack = state.constituents
        for _ in range(self.arity):
            children.append(stack.value)
            stack = stack.pop()
        children.reverse()
        return state.word_position, stack, (self.label, children), CloseConstituent

    def short_name(self):
        return 'REDUCE'

    def short_label(self):
        return 'REDUCE(%s,%d)' % (self.label[0], self.arity)

    def __repr__(self):
        return 'CanonicalBUReduce(%r, %d)' % (self.label[0], self.arity)

    def __eq__(self, other):
        return (isinstance(other, CanonicalBUReduce)
                and self.label == other.label and self.arity == other.arity)

    def __hash__(self):
        return hash(('CanonicalBUReduce', self.label, self.arity))


class CanonicalBUFinish(Transition):
    def is_legal(self, state, model):
        return enabled(model) and not state.finished(model) and root_complete(state, model)

    def update_state(self, state, model):
        return state.word_position, state.constituents, None, None

    def short_name(self):
        return 'FINISH'

    def __repr__(self):
        return 'CanonicalBUFinish'

    def __eq__(self, other):
        return isinstance(other, CanonicalBUFinish)

    def __hash__(self):
        return hash('CanonicalBUFinish')


def yield_canonical_bottom_up_sequence(tree):
    if tree.is_leaf():
        yield CanonicalBUShift()
        return
    if not tree.children:
        raise ValueError('Canonical BU requires nonempty constituents')
    for child in tree.children:
        yield from yield_canonical_bottom_up_sequence(child)
    yield CanonicalBUReduce(tree.label, len(tree.children))
