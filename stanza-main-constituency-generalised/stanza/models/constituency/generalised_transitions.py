"""Delimiter-guided BU/TD/IO, preserving the supplied notebook action strings.

TD's notebook peeks at the next oracle action to determine an NT's role. Here
the marker starts pending; an executed child NT materialises its phrase boundary.
An executed SHIFT leaves it as a POS marker. No gold suffix is consulted.
"""
from dataclasses import dataclass

from stanza.models.constituency.parse_transitions import Transition, TransitionScheme, Dummy, CloseConstituent, too_many_unary_nodes


SCHEMES = {TransitionScheme.GENERALISED_BOTTOM_UP: 'bu',
           TransitionScheme.GENERALISED_TOP_DOWN: 'td',
           TransitionScheme.GENERALISED_IN_ORDER: 'io'}


@dataclass(frozen=True)
class Delimiter:
    label: tuple = ()  # distinct from every nonempty NT label

    def is_preterminal(self):
        return False

    def __format__(self, spec):
        return 'D'

    def __repr__(self):
        return 'D'


@dataclass
class PushSequence:
    nodes: list


def kind(model):
    return SCHEMES.get(model.transition_scheme)


def stack_items(state, model):
    result = []
    stack = state.constituents
    while stack.parent is not None:
        result.append((stack, model.get_top_constituent(stack)))
        stack = stack.pop()
    return list(reversed(result))


def is_tree(item):
    return not isinstance(item, (Dummy, Delimiter))


def root_complete(state, model):
    if state.num_opens or not state.empty_word_queue() or not state.has_one_constituent():
        return False
    tree = model.get_top_constituent(state.constituents)
    return is_tree(tree) and not tree.is_leaf() and tree.label in model.root_labels


def phrase_domain(items, system):
    """Return label-marker index, delimiter index and ordered child indices."""
    if not items:
        return None
    values = [value for _, value in items]
    if system == 'bu':
        marker = len(values) - 1
        if not isinstance(values[marker], Dummy):
            return None
        delimiter = next((index for index in range(marker - 1, -1, -1)
                          if isinstance(values[index], Delimiter)), None)
        children = list(range(0 if delimiter is None else delimiter + 1, marker))
        if children and all(is_tree(values[index]) and not values[index].is_leaf() for index in children):
            return marker, delimiter, children
        return None
    for marker in range(len(values) - 1, -1, -1):
        if not isinstance(values[marker], Dummy):
            continue
        if system == 'td':
            delimiter = marker - 1
            if delimiter < 0 or not isinstance(values[delimiter], Delimiter):
                continue
            children = list(range(marker + 1, len(values)))
        else:
            delimiter = next((index for index in range(marker - 1, -1, -1)
                              if isinstance(values[index], Delimiter)), None)
            if delimiter is None:
                continue
            left = list(range(delimiter + 1, marker))
            if not left:
                continue
            children = left + list(range(marker + 1, len(values)))
        if children and all(is_tree(values[index]) and not values[index].is_leaf() for index in children):
            return marker, delimiter, children
    return None


class GeneralisedAction(Transition):
    op = None

    def short_name(self):
        return self.op

    def __repr__(self):
        return type(self).__name__

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self).__name__)


class GeneralisedShift(GeneralisedAction):
    op = 'SHIFT'
    with_delimiter = False

    def is_legal(self, state, model):
        system = kind(model)
        if system is None or state.finished(model) or state.empty_word_queue() or root_complete(state, model):
            return False
        items = stack_items(state, model)
        if any(is_tree(value) and value.is_leaf() for _, value in items):
            return False
        if system == 'td':
            return (not self.with_delimiter and state.num_opens >= 2 and items
                    and isinstance(items[-1][1], Dummy)
                    and (len(items) < 2 or not isinstance(items[-2][1], Delimiter)))
        if system == 'bu' and items and isinstance(items[-1][1], Dummy):
            return False
        if system == 'io' and not any(isinstance(value, Delimiter) for _, value in items):
            return self.with_delimiter and not items
        return True

    def update_state(self, state, model):
        word = model.transform_word_to_constituent(state)
        if self.with_delimiter:
            return state.word_position + 1, state.constituents, word, GeneralisedDShift
        return state.word_position + 1, state.constituents, word, None


class GeneralisedDShift(GeneralisedShift):
    op = 'D-SHIFT'
    with_delimiter = True

    @staticmethod
    def build_constituents(model, data):
        delimiters = model.dummy_constituents([Delimiter() for _ in data])
        return [PushSequence([delimiter, word]) for delimiter, word in zip(delimiters, data)]


class GeneralisedNT(GeneralisedAction):
    op = 'NT'

    def __init__(self, label):
        if not isinstance(label, str) or not label:
            raise ValueError('GeneralisedNT requires a nonempty label')
        self.label = (label,)

    def delta_opens(self):
        return 1

    def is_legal(self, state, model):
        system = kind(model)
        if system is None or state.finished(model) or root_complete(state, model):
            return False
        items = stack_items(state, model)
        if state.num_opens >= (state.sentence_length + 1) * (model.unary_limit + 2):
            return False
        if system == 'td':
            if state.empty_word_queue() or any(is_tree(value) and value.is_leaf() for _, value in items):
                return False
            if self.label[0] in model.root_labels:
                return state.empty_transitions() and not items
            return state.num_opens > 0
        if not items or not is_tree(items[-1][1]):
            return False
        if items[-1][1].is_leaf():
            return self.label[0] not in model.root_labels
        candidate = items + [(None, Dummy(self.label))]
        domain = phrase_domain(candidate, system)
        if domain is None:
            return False
        _, delimiter, children = domain
        if self.label[0] in model.root_labels:
            return (system == 'io' or state.empty_word_queue()) and (delimiter is None or delimiter == 0)
        if delimiter is None and len(children) == 1 and too_many_unary_nodes(candidate[children[0]][1], model.unary_limit):
            return False
        return True

    def update_state(self, state, model):
        sequence = [Dummy(self.label)]
        stack = state.constituents
        if kind(model) == 'td' and not state.empty_constituents:
            top = model.get_top_constituent(stack)
            if isinstance(top, Dummy):
                # Resolve the existing pending NT as a phrase from this executed
                # child NT, keeping the same surface action inventory.
                stack = stack.pop()
                sequence = [Delimiter(), top, Dummy(self.label)]
        return state.word_position, stack, sequence, GeneralisedNT

    @staticmethod
    def build_constituents(model, data):
        flattened = model.dummy_constituents([node for sequence in data for node in sequence])
        output, offset = [], 0
        for sequence in data:
            output.append(PushSequence(flattened[offset:offset + len(sequence)]))
            offset += len(sequence)
        return output

    def __repr__(self):
        return 'GeneralisedNT(%r)' % self.label[0]

    def __eq__(self, other):
        return isinstance(other, GeneralisedNT) and self.label == other.label

    def __hash__(self):
        return hash(('GeneralisedNT', self.label))


class GeneralisedUReduce(GeneralisedAction):
    op = 'U-REDUCE'

    def delta_opens(self):
        return -1

    def is_legal(self, state, model):
        system = kind(model)
        if system is None or state.finished(model):
            return False
        items = stack_items(state, model)
        if len(items) < 2:
            return False
        marker, child = (items[-2][1], items[-1][1]) if system == 'td' else (items[-1][1], items[-2][1])
        return (isinstance(marker, Dummy) and marker.label[0] not in model.root_labels
                and is_tree(child) and child.is_leaf())

    def update_state(self, state, model):
        stack = state.constituents
        if kind(model) == 'td':
            child = stack.value
            stack = stack.pop()
            label = model.get_top_constituent(stack).label
            stack = stack.pop()
        else:
            label = model.get_top_constituent(stack).label
            stack = stack.pop()
            child = stack.value
            stack = stack.pop()
        return state.word_position, stack, (label, [child]), CloseConstituent


class GeneralisedReduce(GeneralisedAction):
    op = 'REDUCE'
    keep_delimiter = False

    def delta_opens(self):
        return -1

    def is_legal(self, state, model):
        system = kind(model)
        if system is None or state.finished(model) or root_complete(state, model):
            return False
        if self.keep_delimiter and system == 'td':
            return False
        items = stack_items(state, model)
        domain = phrase_domain(items, system)
        if domain is None:
            return False
        marker, delimiter, children = domain
        label = items[marker][1].label[0]
        if self.keep_delimiter and delimiter is None:
            return False
        if label in model.root_labels:
            return (not self.keep_delimiter and state.empty_word_queue()
                    and (delimiter is None or delimiter == 0))
        # IO has no no-delimiter fallback: retain its outer boundary for ROOT.
        if system == 'io' and delimiter == 0 and not self.keep_delimiter:
            return False
        if system != 'td' and len(children) == 1 and (self.keep_delimiter or delimiter is None):
            return not too_many_unary_nodes(items[children[0]][1], model.unary_limit)
        return True

    def update_state(self, state, model):
        items = stack_items(state, model)
        marker, delimiter, indices = phrase_domain(items, kind(model))
        label = items[marker][1].label
        children = [items[index][0].value for index in indices]
        if delimiter is None:
            prefix = items[0][0].pop()
        elif self.keep_delimiter:
            prefix = items[delimiter][0]  # preserve the existing delimiter and neural state
        else:
            prefix = items[delimiter][0].pop()
        return state.word_position, prefix, (label, children), CloseConstituent


class GeneralisedDReduce(GeneralisedReduce):
    op = 'D-REDUCE'
    keep_delimiter = True


class GeneralisedFinish(GeneralisedAction):
    op = 'FINISH'

    def is_legal(self, state, model):
        return kind(model) is not None and not state.finished(model) and root_complete(state, model)

    def update_state(self, state, model):
        return state.word_position, state.constituents, None, None


def build_generalised_sequence(tree, system):
    from stanza.models.constituency.generalised_oracle import get_oracle_bottomup, get_oracle_topdown, get_oracle_inorder
    oracle = {'bu': get_oracle_bottomup, 'td': get_oracle_topdown, 'io': get_oracle_inorder}[system](tree)
    actions = {'SHIFT': GeneralisedShift, 'D-SHIFT': GeneralisedDShift,
               'U-REDUCE': GeneralisedUReduce, 'REDUCE': GeneralisedReduce,
               'D-REDUCE': GeneralisedDReduce, 'FINISH': GeneralisedFinish}
    return [GeneralisedNT(value) if op == 'NT' else actions[op]() for op, value in oracle]
