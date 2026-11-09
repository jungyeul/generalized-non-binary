"""Check executed action histories against the manuscript's structural schemas."""
import argparse
from collections import Counter
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import re
from statistics import mean, stdev

from benchmark_decoding import CORPORA, FAMILIES, ORDERS, SEEDS, write_json


@dataclass(frozen=True)
class Marker:
    label: str
    preterminal: bool


DELIMITER = object()
RELAXED_ACTIONS = ('io_multiple_left_open', 'io_multiple_left_reduce',
                   'bu_no_delimiter_open', 'bu_no_delimiter_reduce')


def decode_tree(text):
    """Read the unretagged decoder output without importing Stanza."""
    pending = []
    tree = None
    for token in re.findall(r'\(|\)|[^\s()]+', text):
        if token == '(':
            pending.append([])
        elif token == ')':
            label, *children = pending.pop()
            node = (label, tuple(children))
            if pending:
                pending[-1].append(node)
            else:
                tree = node
        else:
            pending[-1].append(token)
    if pending or tree is None:
        raise ValueError('Incomplete predicted tree')
    return tree


def leaves(tree):
    if isinstance(tree, str):
        yield tree
    else:
        for child in tree[1]:
            yield from leaves(child)


def replay(actions, target, family, order, roots):
    """Replay successful traces, allowing and counting only the documented relaxations."""
    guided = family == 'generalised'
    words = list(leaves(target))
    stack = []
    counts = Counter()
    label_roles = {'preterminal': set(), 'phrasal': set()}
    position = 0
    finished = False

    def need(condition, message):
        if not condition:
            raise ValueError(f'{family}/{order} action {idx + 1} ({name}): {message}')

    def delimiter_before(end):
        return next((i for i in range(end - 1, -1, -1) if stack[i] is DELIMITER), -1)

    def check_io_left(delimiter, marker, step):
        left_count = marker - delimiter - 1
        need(delimiter >= 0 and left_count >= 1, 'missing in-order left corner')
        need(all(isinstance(item, tuple) for item in stack[delimiter + 1:marker]),
             'in-order left domain contains an unfinished item')
        if left_count > 1:
            counts[f'io_multiple_left_{step}'] += 1

    def compose(start, label, children, preterminal, retain=False):
        if preterminal:
            need(len(children) == 1 and isinstance(children[0], str), 'bad preterminal')
        else:
            need(bool(children) and all(isinstance(child, tuple) for child in children),
                 'bad phrasal child sequence')
        label_roles['preterminal' if preterminal else 'phrasal'].add(label)
        del stack[start + int(retain):]
        # Match Stanza's bracket escaping in the recorded tree string.
        stack.append((label.replace('(', '-LRB-').replace(')', '-RRB-'), tuple(children)))

    for idx, (name, labels, arity) in enumerate(actions):
        need(not finished, 'action after finish')
        label = labels[0] if labels else None
        top = stack[-1] if stack else None
        bare = isinstance(top, str)
        pending_pos = isinstance(top, Marker) and top.preterminal
        if name.endswith('Finish'):
            need(position == len(words) and len(stack) == 1 and isinstance(top, tuple)
                 and top[0] in roots and all(isinstance(child, tuple) for child in top[1]),
                 'invalid final configuration')
            finished = True
        elif name.endswith('Shift'):
            need(position < len(words), 'shift past end')
            need(pending_pos if order == 'td' else not (bare or pending_pos),
                 'shift conflicts with preterminal construction')
            if name == 'GeneralisedDShift':
                need(guided and order != 'td', 'delimiter shift in wrong system')
                stack.append(DELIMITER)
            stack.append(words[position])
            position += 1
        elif name.endswith('NT'):
            need(label is not None and not pending_pos, 'invalid marker')
            if order == 'td':
                need(position < len(words) and not bare and idx + 1 < len(actions),
                     'invalid top-down marker position')
                next_action = actions[idx + 1][0]
                need(next_action.endswith(('Shift', 'NT')), 'unresolved top-down marker')
                preterminal = next_action.endswith('Shift')
                counts['td_deferred_preterminal_marker' if preterminal
                       else 'td_deferred_phrasal_marker'] += 1
                if guided and not preterminal:
                    # The implementation pushes this delimiter with the next NT.
                    # Normalize its placement before checking the formal schema.
                    counts['td_delayed_delimiter'] += 1
                    stack.append(DELIMITER)
            else:
                preterminal = bare
                need(preterminal or isinstance(top, tuple), 'no left child for marker')
                if guided and not preterminal:
                    delimiter = delimiter_before(len(stack))
                    if order == 'io':
                        check_io_left(delimiter, len(stack), 'open')
                    else:
                        need(all(isinstance(item, tuple) for item in stack[delimiter + 1:]),
                             'bottom-up domain contains an unfinished item')
                        if delimiter < 0:
                            counts['bu_no_delimiter_open'] += 1
            stack.append(Marker(label, preterminal))
        elif name == 'CanonicalBUReduce':
            need(family == 'canonical' and order == 'bu' and 1 <= arity <= len(stack),
                 'invalid arity-specific reduction')
            compose(len(stack) - arity, label, stack[-arity:], bare)
        elif name.endswith('Reduce'):
            marker = next((i for i in range(len(stack) - 1, -1, -1)
                           if isinstance(stack[i], Marker)), -1)
            need(marker >= 0, 'no open marker')
            opened = stack[marker]
            retain = name == 'GeneralisedDReduce'
            if opened.preterminal:
                need(not guided or name == 'GeneralisedUReduce', 'wrong preterminal reduction')
                start = marker if order == 'td' else marker - 1
                need(start >= 0 and start + 2 == len(stack), 'unfinished preterminal below stack top')
                compose(start, opened.label, [stack[start + int(order == 'td')]], True)
                continue
            need(name != 'GeneralisedUReduce', 'preterminal action on phrase')
            if not guided:
                start = marker if order == 'td' else marker - 1
                need(start >= 0, 'missing canonical left corner')
                children = stack[start:marker] + stack[marker + 1:]
            else:
                delimiter = delimiter_before(len(stack))
                if order == 'td':
                    need(delimiter >= 0 and delimiter == marker - 1 and not retain,
                         'invalid top-down phrase domain')
                elif order == 'io':
                    check_io_left(delimiter, marker, 'reduce')
                else:
                    need(marker == len(stack) - 1, 'bottom-up marker is not final')
                    if delimiter < 0:
                        need(not retain, 'cannot retain a missing delimiter')
                        counts['bu_no_delimiter_reduce'] += 1
                start = max(delimiter, 0)
                children = stack[delimiter + 1:marker] + stack[marker + 1:]
            compose(start, opened.label, children, False, retain)
        else:
            need(False, 'unknown action')
    if not finished or stack != [target]:
        raise ValueError(f'{family}/{order}: replay failed to reconstruct decoded tree')
    return counts, label_roles


def audit(args):
    runs = []
    for path in sorted(args.traces.glob('*.json.gz')):
        with gzip.open(path, 'rt') as handle:
            trace = json.load(handle)
        run, rows = trace['metadata'], trace['sentences']
        totals, affected = Counter(), Counter()
        role_labels = {'preterminal': set(), 'phrasal': set()}
        sentences = []
        for idx, row in enumerate(rows):
            try:
                counts, roles = replay(row['actions'], decode_tree(row['tree']), run['family'],
                                       run['order'], run['root_labels'])
            except ValueError as error:
                raise ValueError(f'{path.name}, sentence {idx}: {error}') from error
            for role in role_labels:
                role_labels[role].update(roles[role])
            for event, count in counts.items():
                totals[event] += count
                affected[event] += 1
            off_rule = sum(counts[event] for event in RELAXED_ACTIONS)
            if off_rule:
                totals['off_rule'] += off_rule
                affected['off_rule'] += 1
                sentences.append({'index_zero_based': idx, **counts})
        runs.append({**run, 'sentences': len(rows), 'actions': sum(len(row['actions']) for row in rows),
                     'archive_mismatch_indices': [i for i, row in enumerate(rows) if not row['matches_archive']],
                     'counts': dict(totals), 'affected_sentences': dict(affected),
                     'labels_by_role': {role: sorted(labels) for role, labels in role_labels.items()},
                     'off_rule_sentences': sentences})
        print(f"{path.name}: {len(rows)} sentences, {dict(totals)}", flush=True)

    expected = {(family, order, corpus, seed) for family in FAMILIES for order in ORDERS
                for corpus in CORPORA for seed in SEEDS}
    actual = {(r['family'], r['order'], r['corpus'], r['seed']) for r in runs}
    if not actual or len(actual) != len(runs) or (not args.partial and actual != expected):
        raise RuntimeError(f'Missing or duplicated runs: {sorted(expected - actual)}')
    groups = []
    for family in FAMILIES:
        for order in ORDERS:
            for corpus in CORPORA:
                group = sorted((r for r in runs if (r['family'], r['order'], r['corpus']) ==
                                (family, order, corpus)), key=lambda run: run['seed'])
                if len(group) != len(SEEDS):
                    continue
                rates = [100 * r['affected_sentences'].get('off_rule', 0) / r['sentences'] for r in group]
                groups.append({'family': family, 'order': order, 'corpus': corpus,
                               'sentences': sum(r['sentences'] for r in group),
                               'actions': sum(r['actions'] for r in group),
                               'off_rule_actions': sum(r['counts'].get('off_rule', 0) for r in group),
                               'off_rule_sentences': sum(r['affected_sentences'].get('off_rule', 0) for r in group),
                               'per_seed_sentence_rates_percent': rates,
                               'sentence_rate_mean_percent': mean(rates),
                               'sentence_rate_sample_sd_percent': stdev(rates)})
    write_json(args.output, {'groups': groups, 'runs': runs})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--partial', action='store_true')
    audit(parser.parse_args())
