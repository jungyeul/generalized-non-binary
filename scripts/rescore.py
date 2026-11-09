"""Compute the paper's EVALB and default JP-EVALB scores from saved test exports."""
import argparse
import csv
import json
from pathlib import Path
import re
from statistics import mean, stdev
import subprocess
import sys

from benchmark_decoding import CORPORA, FAMILIES, ORDERS, SEEDS, digest, run_directory

ROOT = Path(__file__).resolve().parents[1]


def read_score(text, sentences):
    rows = [line.split() for line in text.splitlines()
            if re.match(r'^\s*\d+\s+\d+\s+\d+\s+[-\d.]', line)]
    rows = [row for row in rows if len(row) == 12]
    if len(rows) != sentences or any(int(row[0]) != i or int(row[2]) != 0
                                     for i, row in enumerate(rows, 1)):
        raise ValueError('Scorer skipped, rejected, or reordered a test sentence')
    matched, gold, predicted = (sum(int(row[i]) for row in rows) for i in (5, 6, 7))
    if not (0 <= matched <= min(gold, predicted)) or gold + predicted == 0:
        raise ValueError('Invalid corpus bracket totals')
    f1 = 200 * matched / (gold + predicted)
    displayed = re.search(r'Bracketing FMeasure\s*=\s*([\d.]+)', text)
    if not displayed or abs(f1 - float(displayed.group(1))) >= .00500001:
        raise ValueError('Corpus F1 disagrees with the displayed scorer result')
    return dict(f1_percent=f1, matched_brackets=matched, gold_brackets=gold,
                predicted_brackets=predicted, sentences=sentences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scorers-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--work', type=Path, default=ROOT)
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=FAMILIES)
    parser.add_argument('--corpora', nargs='+', choices=CORPORA, default=tuple(CORPORA))
    parser.add_argument('--orders', nargs='+', choices=ORDERS, default=ORDERS)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new scoring output directory')

    from nltk import Tree

    tools = args.scorers_dir.resolve()
    repositories = {name: subprocess.check_output(
        ['git', '-C', str(tools / name), 'rev-parse', 'HEAD'], text=True).strip()
        for name in ('EVALB', 'alignment-based-PARSEVAL')}
    ev = tools / 'EVALB' / 'evalb'
    jp = tools / 'alignment-based-PARSEVAL' / 'jp-evalb.py'
    params = jp.parent / 'case1_ptb23_dir' / 'COLLINS.prm'
    scorer_commands = {'evalb': [str(ev), '-p', str(params)],
                       'jp_evalb': [sys.executable, str(jp)]}
    metadata = dict(seeds=list(SEEDS), split='test', standard_deviation='sample (ddof=1)',
                    repositories=repositories, parameter_sha256=digest(params))
    args.output.mkdir(parents=True)
    runs, summary, gold_hashes = [], [], {}
    for family in dict.fromkeys(args.families):
        for order in dict.fromkeys(args.orders):
            for corpus in dict.fromkeys(args.corpora):
                group = []
                for seed in SEEDS:
                    directory = run_directory(args.work.resolve(), family, order, corpus, seed)
                    run = json.loads((directory / 'run.json').read_text())
                    if run['status'] != 'complete' or run['config']['seed'] != seed:
                        raise ValueError(f'Incomplete or mismatched run in {directory}')
                    gold = directory / 'predictions/test.orig.mrg'
                    prediction = directory / 'predictions/test.pred.mrg'
                    gold_hash = digest(gold)
                    if gold_hash != digest(directory / 'data/test.mrg'):
                        raise ValueError(f'Exported reference trees differ in {directory}')
                    if gold_hashes.setdefault(corpus, gold_hash) != gold_hash:
                        raise ValueError(f'Reference trees differ across runs on {corpus}')
                    expected = CORPORA[corpus][1]
                    gold_lines, predicted_lines = [p.read_text(encoding='utf-8').splitlines()
                                                   for p in (gold, prediction)]
                    if len(gold_lines) != expected or len(predicted_lines) != expected:
                        raise ValueError(f'Unexpected test sentence count in {directory}')
                    if any(Tree.fromstring(g).leaves() != Tree.fromstring(p).leaves()
                           for g, p in zip(gold_lines, predicted_lines)):
                        raise ValueError(f'Gold and predicted token yields differ in {directory}')
                    out = args.output / family / order / corpus / str(seed)
                    out.mkdir(parents=True)
                    scores = {}
                    for name, command in scorer_commands.items():
                        result = subprocess.run([*command, str(gold), str(prediction)],
                                                capture_output=True, text=True)
                        (out / f'{name}.stdout').write_text(result.stdout)
                        (out / f'{name}.stderr').write_text(result.stderr)
                        result.check_returncode()
                        scores[name] = read_score(result.stdout, expected)
                    group.append(dict(family=family, order=order, corpus=corpus, seed=seed,
                                      gold_sha256=gold_hash, prediction_sha256=digest(prediction),
                                      scores=scores))
                runs.extend(group)
                for name in scorer_commands:
                    values = [row['scores'][name]['f1_percent'] for row in group]
                    summary.append(dict(family=family, order=order, corpus=corpus, scorer=name,
                                        mean=mean(values), sample_sd=stdev(values)))
    (args.output / 'scores.json').write_text(json.dumps(
        dict(metadata=metadata, runs=runs, summary=summary), indent=2) + '\n')
    for filename, rows in (
        ('summary.csv', summary),
        ('per-seed.csv', [dict(family=r['family'], order=r['order'], corpus=r['corpus'],
                               seed=r['seed'], **{name: score['f1_percent']
                                                for name, score in r['scores'].items()})
                          for r in runs]),
    ):
        with (args.output / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
