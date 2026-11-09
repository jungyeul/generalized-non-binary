"""Canonical non-binary TD from the user's canonical_TD.ipynb.

NT predicts every nonterminal, including POS and ROOT. SHIFT pushes a bare
word. REDUCE supplies neither label nor arity. FINISH is an explicit action.
Stanza's Dummy objects represent the notebook's unmatched LabelMarkers;
neural embeddings/composition are shared, but native TD actions are not used.
"""
from stanza.models.constituency.parse_transitions import (
    Transition, TransitionScheme, Dummy, CloseConstituent,
)


def enabled(model):
    return model.transition_scheme is TransitionScheme.CANONICAL_TOP_DOWN


class CanonicalNT(Transition):
    def __init__(self, label):
        if not isinstance(label, str) or not label:
            raise ValueError('CanonicalNT requires a nonempty label')
        self.label = (label,)

    def delta_opens(self):
        return 1

    def is_legal(self, state, model):
        if not enabled(model) or state.finished(model) or state.empty_word_queue():
            return False
        if not state.empty_constituents:
            top = model.get_top_constituent(state.constituents)
            if not isinstance(top, Dummy) and top.is_leaf():
                return False  # a bare terminal must be reduced immediately
        if self.label[0] in model.root_labels:
            return state.empty_transitions() and state.empty_constituents
        if state.num_opens == 0:
            return False
        # Generous sentence/unary-dependent inference guard. Never collapse
        # a unary chain or impose native TD's two-open closing restriction.
        return state.num_opens < (state.sentence_length + 1) * (model.unary_limit + 2)

    def update_state(self, state, model):
        return state.word_position, state.constituents, Dummy(self.label), CanonicalNT

    @staticmethod
    def build_constituents(model, data):
        return model.dummy_constituents(data)

    def short_name(self):
        return 'NT'

    def __repr__(self):
        return 'CanonicalNT(%r)' % self.label[0]

    def __eq__(self, other):
        return isinstance(other, CanonicalNT) and self.label == other.label

    def __hash__(self):
        return hash(('CanonicalNT', self.label))


class CanonicalShift(Transition):
    def is_legal(self, state, model):
        if state.num_opens == 1:
            # ROOT is the outer wrapper in the experiment data. A bare word
            # directly under ROOT would omit its POS node and could strand
            # unread input; open its leaf-parent first.
            return False
        return (enabled(model) and not state.finished(model)
                and not state.empty_word_queue() and not state.empty_constituents
                and isinstance(model.get_top_constituent(state.constituents), Dummy))

    def update_state(self, state, model):
        return (state.word_position + 1, state.constituents,
                model.transform_word_to_constituent(state), None)

    def short_name(self):
        return 'SHIFT'

    def __repr__(self):
        return 'CanonicalShift'

    def __eq__(self, other):
        return isinstance(other, CanonicalShift)

    def __hash__(self):
        return hash('CanonicalShift')


class CanonicalReduce(Transition):
    def delta_opens(self):
        return -1

    def is_legal(self, state, model):
        if not enabled(model) or state.finished(model) or state.num_opens <= 0:
            return False
        stack = state.constituents
        children = []
        while stack.parent is not None:
            item = model.get_top_constituent(stack)
            if isinstance(item, Dummy):
                break
            children.append(item)
            stack = stack.pop()
        else:
            return False
        if not children:
            return False
        if any(child.is_leaf() for child in children):
            if len(children) != 1 or not children[0].is_leaf():
                return False
        # Closing ROOT early would strand the remaining input.
        if state.num_opens == 1 and not state.empty_word_queue():
            return False
        return True

    def update_state(self, state, model):
        children = []
        stack = state.constituents
        while not isinstance(model.get_top_constituent(stack), Dummy):
            children.append(stack.value)
            stack = stack.pop()
        label = model.get_top_constituent(stack).label
        children.reverse()
        return state.word_position, stack.pop(), (label, children), CloseConstituent

    def short_name(self):
        return 'REDUCE'

    def __repr__(self):
        return 'CanonicalReduce'

    def __eq__(self, other):
        return isinstance(other, CanonicalReduce)

    def __hash__(self):
        return hash('CanonicalReduce')


class CanonicalFinish(Transition):
    def is_legal(self, state, model):
        if not enabled(model) or state.finished(model):
            return False
        if not state.empty_word_queue() or not state.has_one_constituent() or state.num_opens:
            return False
        tree = model.get_top_constituent(state.constituents)
        return not isinstance(tree, Dummy) and not tree.is_leaf() and tree.label in model.root_labels

    def update_state(self, state, model):
        # Explicit no-op on the tree stack; only transition history changes.
        return state.word_position, state.constituents, None, None

    def short_name(self):
        return 'FINISH'

    def __repr__(self):
        return 'CanonicalFinish'

    def __eq__(self, other):
        return isinstance(other, CanonicalFinish)

    def __hash__(self):
        return hash('CanonicalFinish')


def yield_canonical_top_down_sequence(tree):
    if tree.is_leaf() or not tree.children:
        raise ValueError('Canonical TD requires a nonempty parent for each word')
    yield CanonicalNT(tree.label)
    if tree.is_preterminal():
        yield CanonicalShift()
    else:
        for child in tree.children:
            yield from yield_canonical_top_down_sequence(child)
    yield CanonicalReduce()


def canonical_labels(trees):
    labels = set()
    def visit(tree):
        if not tree.is_leaf():
            labels.add(tree.label)
            for child in tree.children:
                visit(child)
    for tree in trees:
        visit(tree)
    return sorted(labels)
