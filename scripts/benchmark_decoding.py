"""Time saved constituency checkpoints on matched test sentences."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

SEEDS = (1111, 2222, 3333)
ORDERS = ('td', 'io', 'bu')
FAMILIES = ('canonical', 'generalised')
BANDS = ((1, 10), (11, 20), (21, 30), (31, 40), (41, None))
SAMPLE_SEED = 20261108
CORPORA = {'ptb': ('en', 2416), 'ctb': ('zh-hans', 348)}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def run_directory(work, family, order, corpus, seed):
    checkout = 'stanza-main-constituency'
    if family == 'generalised':
        checkout += '-generalised'
    return (work / checkout / 'saved_models' / 'constituency'
            / f'{family}_{order}_static' / corpus / f'seed_{seed}')


def sentence_sample(lengths, per_band):
    rng = random.Random(SAMPLE_SEED)
    sample, warmup = [], []
    for low, high in BANDS:
        group = [index for index, length in enumerate(lengths)
                 if low <= length and (high is None or length <= high)]
        rng.shuffle(group)
        if len(group) < per_band + 1:
            raise ValueError(f'Insufficient test sentences in length band {low}--{high}')
        warmup.append(group[0])
        sample.extend(group[1:per_band + 1])
    rng.shuffle(sample)
    return sample, warmup


def affinity_setup(cpu):
    os.sched_setaffinity(0, {cpu})
    os.environ.update(
        CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
    )


def benchmark(args):
    affinity_setup(args.cpu)
    sys.dont_write_bytecode = True
    source = args.work / 'stanza-main-constituency-generalised'
    sys.path.insert(0, str(source))
    import stanza
    import torch
    import transformers
    from stanza.models.constituency.trainer import Trainer
    from stanza.models.constituency.tree_reader import read_treebank
    from stanza.models.constituency.utils import retag_trees
    from stanza.models.common.vocab import VOCAB_PREFIX

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if Path(stanza.__file__).resolve().parent.parent != source.resolve():
        raise RuntimeError(f'Unexpected Stanza source {stanza.__file__}')

    metadata = {
        'cpu': args.cpu,
        'socket': int(Path(f'/sys/devices/system/cpu/cpu{args.cpu}/topology/physical_package_id').read_text()),
        'l3_shared_cpus': Path(f'/sys/devices/system/cpu/cpu{args.cpu}/cache/index3/shared_cpu_list').read_text().strip(),
        'cpu_model': next(line.split(':', 1)[1].strip()
                          for line in Path('/proc/cpuinfo').read_text().splitlines()
                          if line.startswith('model name')),
        'threads': torch.get_num_threads(),
        'interop_threads': torch.get_num_interop_threads(),
        'batch_size': 1, 'device': 'cpu', 'timed_passes': 1,
        'per_band': args.per_band, 'sample_seed': SAMPLE_SEED,
        'seed_order': [args.seed], 'bands': BANDS,
        'versions': {'python': platform.python_version(), 'stanza': stanza.__version__,
                     'torch': torch.__version__, 'transformers': transformers.__version__},
        'source': str(source),
        'source_hashes': {name: digest(source / 'stanza/models/constituency' / name)
                          for name in ('base_model.py', 'lstm_model.py', 'generalised_transitions.py',
                                       'canonical_bottom_up.py', 'canonical_in_order.py',
                                       'canonical_top_down.py', 'trainer.py')},
        'samples': {},
        'timer': 'Wall clock around a native batch-one greedy parse of a pretagged sentence',
        'excluded': ['checkpoint loading', 'POS tagging', 'warmup', 'file I/O', 'evaluation'],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'protocol.json', metadata)

    def decode(model, tree):
        return model.parse_sentences_no_grad(
            iter([tree]), model.build_batch_from_trees, 1, model.predict)

    for corpus, (language, expected_count) in CORPORA.items():
        if corpus != args.corpus:
            continue
        anchor = run_directory(args.work, 'canonical', 'td', corpus, SEEDS[0])
        input_file = anchor / 'data/test.mrg'
        test_hash = digest(input_file)
        trees = read_treebank(str(input_file))
        if len(trees) != expected_count:
            raise RuntimeError(f'Unexpected {corpus} test size {len(trees)}')
        lengths = [len(tree.leaf_labels()) for tree in trees]
        sample, warmup = sentence_sample(lengths, args.per_band)
        selected = warmup + sample
        tagger = stanza.Pipeline(
            lang=language, dir=str(args.resources), processors='tokenize,pos',
            tokenize_pretokenized=True, package={'pos': 'default'}, device='cpu',
            download_method=None, verbose=False)
        if len(tagger.processors['pos'].vocab['xpos']) <= len(VOCAB_PREFIX):
            raise RuntimeError(f'{corpus} tagger has no usable XPOS vocabulary')
        retagged = retag_trees([trees[index] for index in selected], [tagger], xpos=True)
        tagged = dict(zip(selected, retagged))
        metadata['samples'][corpus] = {
            'test_sha256': test_hash, 'test_size': len(trees), 'indices_zero_based': sample,
            'warmup_indices_zero_based': warmup, 'token_counts': [lengths[i] for i in sample],
            'retagged_sample_sha256': hashlib.sha256(
                '\n'.join(str(tree) for tree in retagged).encode()).hexdigest(),
            'pos_model_path': tagger.processors['pos'].config.get('model_path'),
        }
        write_json(args.output / 'protocol.json', metadata)

        for seed_number, seed in enumerate(SEEDS):
            if seed != args.seed:
                continue
            order_cycle = ORDERS[seed_number:] + ORDERS[:seed_number]
            for order in order_cycle:
                families = FAMILIES if seed_number % 2 == 0 else FAMILIES[::-1]
                for family in families:
                    label = f'{family}-{order}-{corpus}-{seed}'
                    directory = run_directory(args.work, family, order, corpus, seed)
                    run = json.loads((directory / 'run.json').read_text())
                    checkpoint = directory / 'models/best.pt'
                    if (run['status'] != 'complete' or run['config']['seed'] != seed
                            or run['config']['system'] != order
                            or run['data']['test']['sha256'] != test_hash
                            or digest(directory / 'data/test.mrg') != test_hash):
                        raise RuntimeError(f'Invalid benchmark inputs for {label}')
                    expected = read_treebank(str(directory / 'predictions/test.pred.mrg'))
                    if len(expected) != len(trees):
                        raise RuntimeError(f'Incomplete saved predictions for {label}')
                    print(f'START {label}', flush=True)
                    trainer = Trainer.load(str(checkpoint), args={'device': torch.device('cpu')},
                                           foundation_cache=tagger.foundation_cache)
                    model = trainer.model
                    model.eval()
                    for index in warmup:
                        if len(decode(model, tagged[index])) != 1:
                            raise RuntimeError(f'Warmup failed for {label}, test sentence {index}')

                    records = []
                    for index in sample:
                        cpu_start = time.process_time_ns()
                        start = time.perf_counter_ns()
                        output = decode(model, tagged[index])
                        elapsed = time.perf_counter_ns() - start
                        cpu_elapsed = time.process_time_ns() - cpu_start
                        if len(output) != 1 or len(output[0].predictions) != 1:
                            raise RuntimeError(f'Decode failed for {label}, test sentence {index}')
                        predicted = output[0].predictions[0].tree
                        if predicted.leaf_labels() != trees[index].leaf_labels():
                            raise RuntimeError(f'Token yield changed for {label}, test sentence {index}')
                        restored = predicted.replace_tags(trees[index])
                        records.append({'index': index, 'tokens': lengths[index],
                                        'elapsed_ns': elapsed, 'cpu_ns': cpu_elapsed,
                                        'matches_saved_tree': restored == expected[index]})
                    total = sum(row['elapsed_ns'] for row in records)
                    result = {'family': family, 'order': order, 'corpus': corpus, 'seed': seed,
                              'checkpoint_sha256': digest(checkpoint), 'sentences': records,
                              'sentences_per_second': len(records) * 1e9 / total}
                    write_json(args.output / f'{label}.json', result)
                    matches = sum(row['matches_saved_tree'] for row in records)
                    median_ms = statistics.median(row['elapsed_ns'] for row in records) / 1e6
                    print(f'DONE {label} {total / 1e9:.2f}s median {median_ms:.2f}ms '
                          f'matches {matches}/{len(records)}', flush=True)
                    del model, trainer
        del tagger
    print('BENCHMARK_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--resources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--corpus', choices=CORPORA, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--per-band', type=int, default=20)
    benchmark(parser.parse_args())
