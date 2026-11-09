"""Aggregate the six corpus–seed timing blocks produced by benchmark_decoding.py."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from benchmark_decoding import BANDS, CORPORA, FAMILIES, ORDERS, SEEDS


def read(path):
    return json.loads(path.read_text())


def summarize(values):
    return {'mean': mean(values), 'sample_sd': stdev(values),
            'seed_values': dict(zip(map(str, SEEDS), values))}


def aggregate(timing_root):
    protocols, runs = {}, {}
    samples = {}
    for corpus in CORPORA:
        for seed in SEEDS:
            label = f'{corpus}-{seed}'
            directory = timing_root / label
            protocol = read(directory / 'protocol.json')
            protocols[label] = protocol
            sample = protocol['samples'][corpus]
            if samples.setdefault(corpus, sample) != sample:
                raise ValueError(f'Benchmark sentences or tags differ across seeds on {corpus}')
            for family in FAMILIES:
                for order in ORDERS:
                    run = read(directory / f'{family}-{order}-{corpus}-{seed}.json')
                    key = (family, order, corpus, seed)
                    if tuple(run[k] for k in ('family', 'order', 'corpus', 'seed')) != key:
                        raise ValueError(f'Timing record does not match its filename: {key}')
                    sentences = run['sentences']
                    if ([r['index'] for r in sentences] != sample['indices_zero_based']
                            or [r['tokens'] for r in sentences] != sample['token_counts']):
                        raise ValueError(f'Mismatched test sample in {key}')
                    runs[key] = run
    groups = {}
    for family in FAMILIES:
        for order in ORDERS:
            for corpus in CORPORA:
                throughput, latency = [], [[] for _ in BANDS]
                for seed in SEEDS:
                    sentences = runs[family, order, corpus, seed]['sentences']
                    throughput.append(1e9 * len(sentences) / sum(r['elapsed_ns'] for r in sentences))
                    for i, (low, high) in enumerate(BANDS):
                        times = [r['elapsed_ns'] / 1e6 for r in sentences
                                 if r['tokens'] >= low and (high is None or r['tokens'] <= high)]
                        if len(times) != protocols[f'{corpus}-{seed}']['per_band']:
                            raise ValueError(f'Unexpected number of samples in band {i + 1}')
                        latency[i].append(mean(times))
                groups[f'{family}-{order}-{corpus}'] = {
                    'throughput': summarize(throughput),
                    'latency_ms': [summarize(values) for values in latency],
                }
    return {'protocol': protocols, 'groups': groups, 'runs': list(runs.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('timing_root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with args.output.open('x') as handle:
        json.dump(aggregate(args.timing_root), handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
