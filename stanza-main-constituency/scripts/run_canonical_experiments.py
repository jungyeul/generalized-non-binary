#!/usr/bin/env python3
"""Shared scheduler: three canonical BU/TD/IO seeds on PTB, aggregate, then CTB.

No model downloads, output writes, or training occur with --dry-run.
Each seed process sees one physical CUDA GPU as logical cuda:0.
"""
import argparse
from copy import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import run_canonical_static as runner

ROOT = Path(__file__).resolve().parents[1]
SEEDS = runner.STANDARD_SEEDS


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def build_parser(system='td'):
    if system not in runner.SYSTEM_SCHEMES:
        raise ValueError('Unknown canonical system: ' + str(system))
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.set_defaults(system=system)
    p.add_argument('--stanza-dir', type=Path, default=ROOT)
    p.add_argument('--data-root', type=Path, default=Path('/home/user/working-dir/penn'))
    p.add_argument('--ptb-data', type=Path)
    p.add_argument('--ctb-data', type=Path)
    p.add_argument('--output-dir', type=Path, default=ROOT / ('saved_models/constituency/canonical_' + system + '_static'))
    p.add_argument('--gpus', nargs=3, type=int, default=[1, 2, 3], metavar=('G1111', 'G2222', 'G3333'))
    p.add_argument('--resources-dir', type=Path, default=Path(os.environ.get('STANZA_RESOURCES_DIR', Path.home() / 'stanza_resources')))
    p.add_argument('--corenlp-home', type=Path)
    p.add_argument('--install-corenlp', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--epoch-size', type=int, default=5000)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--eval-batch-size', type=int, default=32)
    p.add_argument('--hidden-size', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--attention', choices=['none', 'partitioned', 'labeled'], default='none')
    p.add_argument('--finetune-bert', action='store_true')
    p.add_argument('--gold-pos', action='store_true')
    p.add_argument('--charlm', choices=['auto', 'none'], default='auto')
    return p


def validate(args):
    if len(set(args.gpus)) != 3 or any(gpu < 0 for gpu in args.gpus):
        raise ValueError('Provide three distinct nonnegative physical GPU indices')
    for name in ('stanza_dir', 'data_root', 'output_dir', 'resources_dir'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.ptb_data = (args.ptb_data or args.data_root / 'PTB_3').expanduser().resolve()
    args.ctb_data = (args.ctb_data or args.data_root / 'CTB_5.1').expanduser().resolve()
    if args.corenlp_home:
        args.corenlp_home = args.corenlp_home.expanduser().resolve()
    elif os.environ.get('CORENLP_HOME'):
        args.corenlp_home = Path(os.environ['CORENLP_HOME']).expanduser().resolve()
    else:
        args.corenlp_home = args.output_dir / 'corenlp'
        if not args.install_corenlp and not args.dry_run:
            raise ValueError('Provide --corenlp-home, CORENLP_HOME, or --install-corenlp')
    module = runner.SYSTEM_MODULES[args.system]
    if not (args.stanza_dir / ('stanza/models/constituency/' + module + '.py')).is_file():
        raise ValueError('Missing canonical integration in ' + str(args.stanza_dir))
    if args.stanza_dir != ROOT:
        raise ValueError('Run the orchestrator from the source checkout being tested/trained')
    if args.output_dir.exists():
        allowed = {'ptb', 'ctb', 'corenlp', 'batch.json', 'results.md', 'summary.json', '.batch.lock'}
        if any(path.name not in allowed for path in args.output_dir.iterdir()):
            raise ValueError('Output directory contains unrelated files; choose a fresh directory')
        if not args.resume and not args.dry_run and any(
                path.name != 'corenlp' for path in args.output_dir.iterdir()):
            raise ValueError('Existing experiment outputs require --resume or a fresh output directory')
        previous_batch = args.output_dir / 'batch.json'
        if args.resume and previous_batch.is_file():
            previous = json.loads(previous_batch.read_text())
            if previous.get('system', args.system) != args.system:
                raise ValueError('Transition system changed since original batch')
            if previous.get('gpus') != args.gpus:
                raise ValueError('GPU mapping changed since original batch; use the original --gpus')


def seed_arguments(args, dataset, seed):
    language, prefix, directory = ('en', 'en_ptb', args.ptb_data) if dataset == 'ptb' else ('zh-hans', 'zh_ctb', args.ctb_data)
    values = ['--stanza-dir', str(args.stanza_dir), '--output-dir', str(args.output_dir / dataset / ('seed_%d' % seed)),
              '--resources-dir', str(args.resources_dir), '--corenlp-home', str(args.corenlp_home),
              '--seed', str(seed), '--device', 'cuda:0', '--epochs', str(args.epochs),
              '--epoch-size', str(args.epoch_size), '--batch-size', str(args.batch_size),
              '--eval-batch-size', str(args.eval_batch_size), '--hidden-size', str(args.hidden_size),
              '--learning-rate', str(args.learning_rate), '--attention', args.attention, '--charlm', args.charlm]
    for split in ('train', 'dev', 'test'):
        values += ['--' + split, str(directory / (prefix + '_' + split + '.mrg'))]
    if args.finetune_bert:
        values.append('--finetune-bert')
    if args.gold_pos:
        values.append('--gold-pos')
    child = runner.build_parser(language, args.system).parse_args(values)
    return language, values, child


def config_dict(child):
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(child).items()
            if key not in ('resume', 'evaluate_only', 'dry_run', 'install_corenlp')}


def source_hashes(child):
    return {name: runner.sha256(child.stanza_dir / name) for name in runner.source_paths(child)}


def validate_seed(child, resume, dry_run=False):
    runner.validate_options(child)
    files = runner.source_files(child)
    manifest = child.output_dir / 'run.json'
    if not manifest.exists():
        if child.output_dir.exists() and any(child.output_dir.iterdir()):
            raise ValueError('Seed directory has files but no run.json: ' + str(child.output_dir))
        return 'new'
    if not resume and not dry_run:
        raise ValueError('Existing seed requires --resume: ' + str(child.output_dir))
    previous = json.loads(manifest.read_text())
    if previous['config'] != config_dict(child) or previous['source_hashes'] != source_hashes(child):
        raise ValueError('Source or settings changed since original run: ' + str(manifest))
    for split, paths in files.items():
        observed = [{'path': str(path), 'sha256': runner.sha256(path)} for path in paths]
        if previous['data'][split]['sources'] != observed:
            raise ValueError('Input changed since original run: ' + str(manifest))
    if previous['status'] == 'complete':
        for split in ('dev', 'test'):
            result = previous['results'][split]
            if not math.isfinite(result['f1']) or not 0 <= result['f1'] <= 1 or result['sentences'] <= 0:
                raise ValueError('Invalid completed metrics: ' + str(manifest))
            for name in ('predictions', 'log'):
                if not Path(result[name]).is_file():
                    raise FileNotFoundError(result[name])
        if not (child.output_dir / 'models/best.pt').is_file():
            raise FileNotFoundError(str(child.output_dir / 'models/best.pt'))
        return 'complete'
    if not (child.output_dir / 'models/checkpoint.pt').is_file():
        raise FileNotFoundError('Cannot resume without checkpoint: ' + str(child.output_dir))
    return 'resume'


def stop_workers(workers):
    for process, _ in workers:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process, _ in workers:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_parallel(args, dataset, states):
    workers, streams = [], []
    try:
        for seed, gpu in zip(SEEDS, args.gpus):
            language, values, child = seed_arguments(args, dataset, seed)
            mode = states[(dataset, seed)]
            if mode == 'complete':
                print('%s seed %d: reusing verified completed run' % (dataset.upper(), seed), flush=True)
                continue
            if mode == 'resume':
                values.append('--resume')
            script = 'run_canonical_' + args.system + ('_ptb.py' if language == 'en' else '_ctb.py')
            log = child.output_dir.parent / ('seed_%d.console.log' % seed)
            stream = log.open('a' if mode == 'resume' else 'w')
            streams.append(stream)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
            env['PYTHONPATH'] = str(args.stanza_dir) + os.pathsep + env.get('PYTHONPATH', '')
            process = subprocess.Popen([sys.executable, str(args.stanza_dir / 'scripts' / script), *values],
                cwd=args.stanza_dir, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            workers.append((process, seed))
            print('%s seed %d -> physical GPU %d (logical cuda:0); log %s' % (dataset.upper(), seed, gpu, log), flush=True)
        pending = list(workers)
        while pending:
            for process, seed in list(pending):
                status = process.poll()
                if status is None:
                    continue
                pending.remove((process, seed))
                if status:
                    raise RuntimeError('%s seed %d failed (exit %d); remaining workers will be stopped' % (dataset.upper(), seed, status))
                print('%s seed %d completed' % (dataset.upper(), seed), flush=True)
            if pending:
                time.sleep(0.25)
    except BaseException:
        stop_workers(workers)
        raise
    finally:
        for stream in streams:
            stream.close()


def aggregate_dataset(args, dataset):
    directory = args.output_dir / dataset
    runs = [json.loads((directory / ('seed_%d/run.json' % seed)).read_text()) for seed in SEEDS]
    summary = runner.summarize_runs(runs)
    if len({json.dumps(run['source_hashes'], sort_keys=True) for run in runs}) != 1:
        raise ValueError('Source hashes differ between seeds')
    for split in ('train', 'dev', 'test'):
        if len({json.dumps(run['data'][split], sort_keys=True) for run in runs}) != 1:
            raise ValueError('Prepared data differ between seeds: ' + split)
    for run, gpu in zip(runs, args.gpus):
        if run['runtime']['cuda_visible_devices'] != str(gpu) or run['runtime']['device'] != 'cuda:0':
            raise ValueError('Seed runtime does not match its assigned GPU')
    summary['gpu_mapping'] = dict(zip(map(str, SEEDS), args.gpus))
    atomic_json(directory / 'summary.json', summary)
    runner.write_results_markdown(directory / 'results.md', summary)
    for split in ('dev', 'test'):
        result = summary['aggregate'][split]
        print('%s %s F1: %.2f +/- %.2f (sample SD, n=3)' % (
            dataset.upper(), split, result['mean_f1_percent'], result['sample_sd_f1_percent']), flush=True)
    return summary


def prepare_runtime(args):
    # Do shared downloads/preflight once before launching concurrent workers.
    # All child environments isolate their POS tagger and parser on one GPU.
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))
    sys.path.insert(0, str(args.stanza_dir))
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 3:
        raise RuntimeError('Three requested CUDA GPUs must be visible for this experiment')
    for dataset in ('ptb', 'ctb'):
        _, _, child = seed_arguments(args, dataset, SEEDS[0])
        # Install the shared CoreNLP once, then use its resolved location.
        if not list(args.corenlp_home.glob('stanford-corenlp-*.jar')):
            if not args.install_corenlp:
                raise RuntimeError('CoreNLP installation missing at ' + str(args.corenlp_home))
            import stanza
            stanza.install_corenlp(dir=str(args.corenlp_home))
        runner.runtime_preflight(child)
        runner.resource_paths(child, download=True)
        if not child.gold_pos:
            import stanza
            stanza.download(child.lang, model_dir=str(child.resources_dir), processors='pos', package=child.retag_package, verbose=False)
        from transformers import AutoModel, AutoTokenizer
        AutoTokenizer.from_pretrained(child.bert_model)
        model = AutoModel.from_pretrained(child.bert_model)
        del model


def main(argv=None, system='td'):
    args = build_parser(system).parse_args(argv)
    validate(args)
    states = {}
    # Check paths/configuration for BOTH datasets before starting PTB.
    for dataset in ('ptb', 'ctb'):
        for seed, gpu in zip(SEEDS, args.gpus):
            _, values, child = seed_arguments(args, dataset, seed)
            # CoreNLP need not exist during a dry run or before installation.
            home = child.corenlp_home
            child.corenlp_home = None if not home.exists() else home
            runner.validate_options(child)
            runner.source_files(child)
            child.corenlp_home = home
            if args.dry_run:
                print('%s seed %d: CUDA_VISIBLE_DEVICES=%d' % (dataset.upper(), seed, gpu))
                for stage, command in runner.commands(child, runner.resource_paths(child)).items():
                    print('[%s] %s' % (stage, shlex.join(command)))
            else:
                # validate_seed handles the final installation path after preflight.
                states[(dataset, seed)] = None
        if args.dry_run:
            print('Barrier: aggregate all three %s seeds (mean/sample SD) before continuing.\n' % dataset.upper())
    if args.dry_run:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Reject simultaneous orchestrators sharing the same experiment directory.
    import fcntl
    with (args.output_dir / '.batch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another canonical batch is using this output directory')
        batch = {'status': 'preflight', 'started_utc': utc_now(), 'seeds': list(SEEDS),
                 'gpus': args.gpus, 'system': args.system, 'experiment_role': runner.SYSTEM_ROLES[args.system], 'dataset_order': ['ptb', 'ctb'], 'completed_datasets': []}
        batch_file = args.output_dir / 'batch.json'
        try:
            prepare_runtime(args)
            for dataset in ('ptb', 'ctb'):
                for seed in SEEDS:
                    _, _, child = seed_arguments(args, dataset, seed)
                    states[(dataset, seed)] = validate_seed(child, args.resume)
                (args.output_dir / dataset).mkdir(exist_ok=True)
            # Clear aggregate results; each dataset is revalidated and aggregated.
            for path in [args.output_dir / 'summary.json', args.output_dir / 'results.md'] + [
                    args.output_dir / dataset / name for dataset in ('ptb', 'ctb') for name in ('summary.json', 'results.md')]:
                path.unlink(missing_ok=True)
            summaries = {}
            for dataset in ('ptb', 'ctb'):
                batch['status'], batch['active_dataset'] = 'running', dataset
                atomic_json(batch_file, batch)
                run_parallel(args, dataset, states)
                summaries[dataset] = aggregate_dataset(args, dataset)
                batch['completed_datasets'].append(dataset)
                atomic_json(batch_file, batch)
            atomic_json(args.output_dir / 'summary.json', summaries)
            lines = ['# Canonical %s static-oracle results' % args.system.upper(), '',
                     'Experiment role: %s.' % runner.SYSTEM_ROLES[args.system], '',
                     'Seeds: 1111, 2222, 3333. Standard deviation: sample (ddof=1).', '',
                     '| Dataset | Dev F1 | Test F1 |', '|---|---:|---:|']
            for dataset, summary in summaries.items():
                scores = [summary['aggregate'][split] for split in ('dev', 'test')]
                lines.append('| %s | %.2f ± %.2f | %.2f ± %.2f |' % (dataset.upper(),
                    scores[0]['mean_f1_percent'], scores[0]['sample_sd_f1_percent'],
                    scores[1]['mean_f1_percent'], scores[1]['sample_sd_f1_percent']))
            lines += ['', 'Individual seeds: [PTB](ptb/results.md), [CTB](ctb/results.md).', '']
            (args.output_dir / 'results.md').write_text('\n'.join(lines))
            batch['status'] = 'complete'
            batch['completed_utc'] = utc_now()
            return 0
        except BaseException as error:
            batch['status'], batch['error'] = 'failed', str(error) or type(error).__name__
            raise
        finally:
            atomic_json(batch_file, batch)


def interrupted(signum, frame):
    raise KeyboardInterrupt('Received termination signal %d' % signum)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as error:
        print('Error: ' + str(error), file=sys.stderr)
        sys.exit(1)
