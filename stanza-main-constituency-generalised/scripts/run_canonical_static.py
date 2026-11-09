#!/usr/bin/env python3
"""Shared train/dev/test runner for canonical and delimiter-guided BU, TD and IO.

Run --help for all options.  --dry-run resolves inputs and prints commands
without importing Stanza, downloading models, or starting training.
"""
from __future__ import annotations

import argparse
from copy import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import statistics
import sys

STANDARD_SEEDS = (1111, 2222, 3333)

LANGUAGE_PROFILES = {
    'en': {'shorthand': 'en_wsj_c', 'bert_model': 'google-bert/bert-large-cased',
           'pretrain': 'conll17', 'charlm': '1billion', 'name': 'English PTB'},
    'zh-hans': {'shorthand': 'zh-hans_ctb-51_c', 'bert_model': 'google-bert/bert-base-chinese',
                'pretrain': 'fasttext157', 'charlm': 'gigaword', 'name': 'Chinese CTB 5.1'},
}

SPLITS = {'train': range(2, 22), 'dev': [22], 'test': [23]}
PROTECTED = {'mode', 'train_file', 'eval_file', 'transition_scheme', 'oracle_level',
             'oracle_frequency', 'oracle_forced_errors', 'load_name', 'load_package',
             'save_dir', 'save_name', 'checkpoint_save_name', 'predict_dir',
             'predict_file', 'num_generate', 'reversed', 'finetune', 'relearn_structure',
             'limit_treebank', 'no_check_valid_states', 'retag_package', 'no_retag',
             'use_peft', 'wordvec_pretrain_file', 'charlm_forward_file',
             'charlm_backward_file', 'retag_method', 'retag_model_path',
             'retag_pretrain_path', 'retag_charlm_forward_file', 'retag_charlm_backward_file',
             'bert_model', 'no_bert_model', 'bert_hidden_layers', 'no_checkpoint', 'seed',
             'lang', 'shorthand', 'device', 'cpu', 'cuda'}


SYSTEM_SCHEMES = {'bu': 'CANONICAL_BOTTOM_UP', 'td': 'CANONICAL_TOP_DOWN', 'io': 'CANONICAL_IN_ORDER'}
GENERALISED_SCHEMES = {'bu': 'GENERALISED_BOTTOM_UP', 'td': 'GENERALISED_TOP_DOWN', 'io': 'GENERALISED_IN_ORDER'}
VARIANTS = {'canonical': SYSTEM_SCHEMES, 'generalised': GENERALISED_SCHEMES}


def scheme_name(args):
    return VARIANTS[args.variant][args.system]


SYSTEM_ROLES = {'bu': 'paper', 'td': 'paper', 'io': 'future_reference'}
SYSTEM_NAMES = {'bu': 'bottom-up', 'td': 'top-down', 'io': 'in-order'}
SYSTEM_MODULES = {'bu': 'canonical_bottom_up', 'td': 'canonical_top_down', 'io': 'canonical_in_order'}


def build_parser(language='en', system='td', variant='canonical'):
    if system not in SYSTEM_SCHEMES:
        raise ValueError('Unknown transition system: ' + str(system))
    if variant not in VARIANTS:
        raise ValueError('Unknown parser variant: ' + str(variant))
    profile = LANGUAGE_PROFILES[language]
    p = argparse.ArgumentParser(description=__doc__ + '\nDataset: ' + profile['name'],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.set_defaults(lang=language, system=system, variant=variant)
    p.add_argument('--ptb-dir', type=Path, help='WSJ directory with sections 02..23 (recursive *.mrg files)')
    for split in SPLITS:
        p.add_argument('--' + split, type=Path, help='Prepared bracketed constituency tree file')
    p.add_argument('--output-dir', type=Path, required=True, help='Separate output directory for this experiment')
    p.add_argument('--stanza-dir', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--bert-model', default=profile['bert_model'], help='Hugging Face identifier or local transformer directory')
    p.add_argument('--finetune-bert', action='store_true', help='Full transformer fine-tuning (more GPU memory)')
    p.add_argument('--bert-lr-scale', type=float, default=0.02, help='Fine-tuning LR multiplier: learning-rate * this value')
    p.add_argument('--bert-hidden-layers', type=int, default=4)
    p.add_argument('--charlm', choices=['auto', 'none'], default='auto', help='Auto uses %s forward/backward %s models' % (language, profile['charlm']))
    p.add_argument('--charlm-forward', type=Path)
    p.add_argument('--charlm-backward', type=Path)
    p.add_argument('--pretrain', type=Path, help='Stanza word-vector .pt file; default downloads the %s pretrain' % language)
    p.add_argument('--resources-dir', type=Path, default=Path(os.environ.get('STANZA_RESOURCES_DIR', Path.home() / 'stanza_resources')))
    p.add_argument('--retag-package', default='default', help='Stanza %s POS package, consistently used for train/dev/test' % language)
    p.add_argument('--retag-model', type=Path, help='Optional local POS model')
    p.add_argument('--gold-pos', action='store_true', help='Use treebank POS tags for all splits; label results as gold-POS')
    p.add_argument('--attention', choices=['none', 'partitioned', 'labeled'], default='none')
    p.add_argument('--epochs', type=int, default=200, help='Stanza iterations; each samples epoch-size trees')
    p.add_argument('--epoch-size', type=int, default=5000, help='0 = entire training split per iteration')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--eval-batch-size', type=int, default=32)
    p.add_argument('--hidden-size', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--seed', type=int, choices=STANDARD_SEEDS, default=None,
                   help='Run one standard seed only; default runs 1111, 2222, and 3333 sequentially')
    p.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0, or another PyTorch device')
    p.add_argument('--corenlp-home', type=Path, help='CoreNLP installation; also accepts CORENLP_HOME environment variable')
    p.add_argument('--install-corenlp', action='store_true', help='Download CoreNLP into output-dir/corenlp when needed (Java required)')
    p.add_argument('--resume', action='store_true', help='Resume this exact experiment from its optimizer checkpoint')
    p.add_argument('--evaluate-only', action='store_true', help='Evaluate this experiment\'s existing best model without training')
    p.add_argument('--dry-run', action='store_true', help='Print the command plan; no writes or model downloads')
    p.add_argument('--stanza-args', nargs=argparse.REMAINDER, default=[], help='Additional native TRAINING options, placed last')
    return p


def validate_options(args):
    if args.lang not in LANGUAGE_PROFILES:
        raise ValueError('Unsupported language: ' + args.lang)
    if args.lang != 'en' and args.ptb_dir:
        raise ValueError('CTB requires prepared --train/--dev/--test files; --ptb-dir uses English WSJ sections')
    supplied = [getattr(args, name) for name in SPLITS]
    if args.ptb_dir and any(supplied):
        raise ValueError('Use --ptb-dir OR all three --train/--dev/--test files')
    if not args.ptb_dir and not all(supplied):
        raise ValueError('Provide --ptb-dir or all three --train/--dev/--test files')
    if args.resume and args.evaluate_only:
        raise ValueError('--resume and --evaluate-only are mutually exclusive')
    if bool(args.charlm_forward) != bool(args.charlm_backward):
        raise ValueError('Provide both --charlm-forward and --charlm-backward')
    if args.charlm == 'none' and args.charlm_forward:
        raise ValueError('Custom character models conflict with --charlm none')
    if args.gold_pos and args.retag_model:
        raise ValueError('--gold-pos conflicts with --retag-model')
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.hidden_size, args.bert_hidden_layers) <= 0 or args.epoch_size < 0:
        raise ValueError('Epochs, batch sizes, dimensions and layer count must be positive; epoch-size must be >= 0')
    if args.learning_rate <= 0 or args.bert_lr_scale <= 0:
        raise ValueError('Learning rates must be positive')
    for token in args.stanza_args:
        if token.startswith('--') and token[2:].split('=', 1)[0] in PROTECTED:
            raise ValueError('This option is controlled by the wrapper: ' + token)
    for name in ['stanza_dir', 'output_dir', 'resources_dir']:
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not (args.stanza_dir / 'stanza/models/constituency/parse_transitions.py').is_file():
        raise ValueError('Not a Stanza source directory: ' + str(args.stanza_dir))
    for name in ['ptb_dir', 'train', 'dev', 'test', 'pretrain', 'charlm_forward', 'charlm_backward', 'retag_model', 'corenlp_home']:
        path = getattr(args, name)
        if path:
            path = path.expanduser().resolve()
            setattr(args, name, path)
            if not path.exists():
                raise FileNotFoundError(str(path))
            expected_dir = name in ('ptb_dir', 'corenlp_home')
            if expected_dir and not path.is_dir():
                raise ValueError('Expected a directory: ' + str(path))
            if not expected_dir and not path.is_file():
                raise ValueError('Expected a file: ' + str(path))


def source_files(args):
    if not args.ptb_dir:
        result = {name: [getattr(args, name)] for name in SPLITS}
    else:
        result = {}
        for name, sections in SPLITS.items():
            files = []
            for section in sections:
                directory = args.ptb_dir / ('%02d' % section)
                section_files = sorted(directory.rglob('*.mrg'))
                if not section_files:
                    raise FileNotFoundError('No .mrg files in WSJ section ' + str(directory))
                files.extend(section_files)
            result[name] = files
    flat = [p for paths in result.values() for p in paths]
    if len(flat) != len(set(flat)):
        raise ValueError('Train/dev/test share an input file')
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_data(files, directory):
    from stanza.models.constituency.parse_tree import Tree
    from stanza.models.constituency.tree_reader import read_tree_file
    directory.mkdir(parents=True, exist_ok=True)
    report = {}
    for split, paths in files.items():
        count = 0
        excluded = 0
        target = directory / (split + '.mrg')
        with target.open('w', encoding='utf-8') as output:
            for path in paths:
                for tree in read_tree_file(str(path)):
                    tree = tree.prune_none()
                    if tree is None or tree.is_leaf():
                        excluded += 1
                        continue
                    tree = tree.simplify_labels()
                    if tree.label not in ('ROOT', 'TOP'):
                        tree = Tree('ROOT', [tree])
                    else:
                        tree = Tree('ROOT', tree.children)
                    if len(tree.children) != 1:
                        raise ValueError('Expected one child of ROOT in %s: %s' % (path, tree))
                    if not list(tree.yield_preterminals()):
                        raise ValueError('No POS-tagged terminals in ' + str(path))
                    output.write(str(tree) + '\n')
                    count += 1
        if not count:
            raise ValueError('Empty prepared split: ' + split)
        report[split] = {'sentences': count, 'excluded_empty': excluded, 'sha256': sha256(target),
                         'sources': [{'path': str(p), 'sha256': sha256(p)} for p in paths]}
    return report


def resource_paths(args, download=False):
    profile = LANGUAGE_PROFILES[args.lang]
    if not download:
        pretrain = args.pretrain or args.resources_dir / args.lang / 'pretrain' / (profile['pretrain'] + '.pt')
        forward = args.charlm_forward or args.resources_dir / args.lang / 'forward_charlm' / (profile['charlm'] + '.pt')
        backward = args.charlm_backward or args.resources_dir / args.lang / 'backward_charlm' / (profile['charlm'] + '.pt')
    else:
        from stanza.resources.default_packages import default_pretrains, default_charlms
        from stanza.utils.training.common import find_wordvec_pretrain, find_charlm_file
        pretrain = args.pretrain or Path(find_wordvec_pretrain(args.lang, default_pretrains, model_dir=str(args.resources_dir)))
        forward = backward = None
        if args.charlm != 'none':
            forward = args.charlm_forward or Path(find_charlm_file('forward', args.lang, default_charlms[args.lang], model_dir=str(args.resources_dir)))
            backward = args.charlm_backward or Path(find_charlm_file('backward', args.lang, default_charlms[args.lang], model_dir=str(args.resources_dir)))
    values = ['--wordvec_pretrain_file', str(pretrain)]
    if args.charlm != 'none':
        values += ['--charlm_forward_file', str(forward), '--charlm_backward_file', str(backward)]
    return values


def commands(args, resources):
    data = args.output_dir / 'data'
    models = args.output_dir / 'models'
    predictions = args.output_dir / 'predictions'
    common = ['--lang', args.lang, '--shorthand', LANGUAGE_PROFILES[args.lang]['shorthand'][:-1] + ('c' if args.variant == 'canonical' else 'g') + args.system, '--transition_scheme', scheme_name(args),
              '--oracle_level', '0', '--oracle_frequency', '0', '--oracle_forced_errors', '0',
              '--save_dir', str(models), '--save_name', 'best.pt',
              '--retag_method', 'xpos', '--eval_batch_size', str(args.eval_batch_size),
              '--seed', str(args.seed)] + resources
    if args.device != 'auto':
        common += ['--device', args.device]
    if args.gold_pos:
        common += ['--no_retag']
    else:
        common += ['--retag_package', args.retag_package]
        if args.retag_model:
            common += ['--retag_model_path', str(args.retag_model)]
    entry = [sys.executable, '-m', 'stanza.models.constituency_parser']
    train = entry + common + [
        '--mode', 'train', '--train_file', str(data / 'train.mrg'), '--eval_file', str(data / 'dev.mrg'),
        '--bert_model', args.bert_model, '--bert_hidden_layers', str(args.bert_hidden_layers),
        '--no_multistage', '--optim', 'adamw', '--learning_rate', str(args.learning_rate),
        '--loss', 'cross', '--epochs', str(args.epochs), '--epoch_size', str(args.epoch_size),
        '--train_batch_size', str(args.batch_size), '--hidden_size', str(args.hidden_size),
        '--checkpoint_save_name', str(models / 'checkpoint.pt')]
    if args.finetune_bert:
        train += ['--bert_finetune', '--bert_learning_rate', str(args.bert_lr_scale)]
    else:
        train += ['--no_bert_finetune']
    if args.attention != 'none':
        train += ['--pattn_num_layers', '2', '--pattn_d_model', '512']
    if args.attention == 'labeled':
        train += ['--use_lattn']
    train += args.stanza_args
    plan = {} if args.evaluate_only else {'train': train}
    for split in ('dev', 'test'):
        plan[split] = entry + common + [
            '--mode', 'predict', '--load_name', str(models / 'best.pt'),
            '--eval_file', str(data / (split + '.mrg')), '--predict_dir', str(predictions),
            '--predict_file', split, '--predict_output_gold_tags']
    return plan


def run_stage(name, command, args, env):
    logfile = args.output_dir / 'logs' / (name + '.log')
    print('\n[%s] %s' % (name, shlex.join(command)), flush=True)
    with logfile.open('w', encoding='utf-8') as log:
        with subprocess.Popen(command, cwd=args.stanza_dir, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
            returncode = process.wait()
    if returncode:
        raise RuntimeError('%s failed (exit %d); see %s' % (name, returncode, logfile))
    return logfile


def evaluation_result(logfile, prepared, predictions):
    from stanza.models.constituency.tree_reader import read_tree_file
    gold = read_tree_file(str(prepared))
    pred = read_tree_file(str(predictions))
    if len(pred) != len(gold):
        raise RuntimeError('Incomplete evaluation: %d/%d predicted trees in %s' % (len(pred), len(gold), predictions))
    for i, (g, p) in enumerate(zip(gold, pred), 1):
        if g.leaf_labels() != p.leaf_labels():
            raise RuntimeError('Prediction sentence alignment mismatch at sentence %d in %s' % (i, predictions))
    scores = re.findall(r'F1 score on .*?:\s*([0-9]+(?:\.[0-9]+)?)', logfile.read_text())
    if not scores:
        raise RuntimeError('Missing Stanza F1 result in ' + str(logfile))
    f1 = float(scores[-1])
    if not 0 <= f1 <= 1:
        raise RuntimeError('Unexpected Stanza F1 scale: ' + str(f1))
    return {'f1': f1, 'f1_percent': 100 * f1, 'sentences': len(gold), 'predictions': str(predictions), 'log': str(logfile)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def validate_native_plan(plan):
    from stanza.models import constituency_parser
    # Native argparse accepts abbreviations; reject ambiguous overrides in
    # passthrough options before any training or resource downloads.
    parser = constituency_parser.build_argparse()
    parser.allow_abbrev = False
    for command in plan.values():
        parser.parse_args(command[3:])


def runtime_preflight(args):
    import torch
    from stanza.models.constituency.parse_transitions import TransitionScheme
    if not hasattr(TransitionScheme, scheme_name(args)):
        raise RuntimeError('This Stanza checkout does not contain the ' + args.variant + ' ' + args.system.upper() + ' integration')
    for package in ['transformers', 'numpy']:
        if importlib.util.find_spec(package) is None:
            raise RuntimeError('Missing dependency %s; install this Stanza checkout with its transformers extra' % package)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested, but this PyTorch environment cannot use CUDA')
    if args.device.startswith('cuda'):
        requested = torch.device(args.device)
        if requested.index is not None and requested.index >= torch.cuda.device_count():
            raise RuntimeError('Requested %s, but only %d CUDA device(s) are visible' % (
                args.device, torch.cuda.device_count()))
    java = shutil.which('java')
    if not java:
        raise RuntimeError('Java is required for CoreNLP bracket evaluation')
    check = subprocess.run([java, '-version'], capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError('Java is installed but cannot start: ' + check.stderr.strip())
    home = args.corenlp_home or (Path(os.environ['CORENLP_HOME']) if os.environ.get('CORENLP_HOME') else None)
    if home is None and args.install_corenlp:
        import stanza
        home = args.output_dir / 'corenlp'
        if not list(home.glob('stanford-corenlp-*.jar')):
            stanza.install_corenlp(dir=str(home))
    if home:
        os.environ['CORENLP_HOME'] = str(home)
    # Send a real request to ensure the scoring class and Java runtime work.
    from stanza.models.constituency.tree_reader import read_trees
    from stanza.server.parser_eval import EvaluateParser, ParseResult, ScoredTree
    tree = read_trees('(ROOT (S (NN preflight)))')[0]
    with EvaluateParser() as evaluator:
        result = evaluator.process([ParseResult(tree, [ScoredTree(tree, 0.0)], None, None)])
    if abs(result.f1 - 1.0) > 1e-6:
        raise RuntimeError('CoreNLP evaluator did not pass its self-check')
    import stanza
    import transformers
    return {'python': sys.version, 'torch': torch.__version__, 'transformers': transformers.__version__,
            'stanza': stanza.__version__, 'cuda_available': torch.cuda.is_available(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'device': args.device,
            'java': (check.stderr or check.stdout).strip(), 'corenlp_home': str(home) if home else None}


def source_paths(args):
    names = ['stanza/models/constituency/' + name + '.py' for name in
             ('parse_transitions', 'transition_sequence', 'parser_training', 'base_model',
              'lstm_model', 'state', 'utils', 'trainer', 'parse_tree',
              'canonical_top_down', 'canonical_bottom_up', 'canonical_in_order',
              'generalised_oracle', 'generalised_transitions')]
    names += ['stanza/models/constituency_parser.py',
              'scripts/run_canonical_static.py', 'scripts/run_canonical_experiments.py',
              'scripts/run_' + args.variant + '_' + args.system + '_ptb.py',
              'scripts/run_' + args.variant + '_' + args.system + '_experiments.py']
    if args.lang == 'zh-hans':
        names.append('scripts/run_' + args.variant + '_' + args.system + '_ctb.py')
    return names


def run_single(args):
    validate_options(args)
    files = source_files(args)
    plan = commands(args, resource_paths(args))
    if args.dry_run:
        for split, paths in files.items():
            print('%s: %d source file(s)' % (split, len(paths)))
        for name, command in plan.items():
            print('[%s] %s' % (name, shlex.join(command)))
        return 0
    sys.path.insert(0, str(args.stanza_dir))
    os.environ['STANZA_RESOURCES_DIR'] = str(args.resources_dir)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ('resume', 'evaluate_only', 'dry_run', 'install_corenlp')}
    source_hashes = {name: sha256(args.stanza_dir / name) for name in source_paths(args)}
    manifest_file = args.output_dir / 'run.json'
    previous = None
    if args.resume or args.evaluate_only:
        if not manifest_file.is_file():
            raise ValueError('No existing run.json in output-dir; use a fresh run first')
        previous = json.loads(manifest_file.read_text())
        if previous.get('source_hashes') != source_hashes:
            raise ValueError('Stanza source changed since this experiment; use a new output directory')
        if previous['config'] != config:
            raise ValueError('Resume/evaluation configuration differs from run.json; use the original options')
        expected = args.output_dir / 'models' / ('best.pt' if args.evaluate_only else 'checkpoint.pt')
        if not expected.is_file():
            raise FileNotFoundError(str(expected))
    elif args.output_dir.exists() and any(
            p.name != 'corenlp' or not args.install_corenlp for p in args.output_dir.iterdir()):
        raise ValueError('Output directory is not empty; use a fresh directory or --resume')
    # Hash inputs before permitting a resumed experiment to replace its data.
    if previous:
        for split, paths in files.items():
            observed = [{'path': str(p), 'sha256': sha256(p)} for p in paths]
            if observed != previous['data'][split]['sources']:
                raise ValueError('Input data changed since original run: ' + split)
    validate_native_plan(plan)
    runtime = runtime_preflight(args)
    for directory in ['models', 'logs', 'predictions']:
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)
    data = prepare_data(files, args.output_dir / 'data')
    plan = commands(args, resource_paths(args, download=True))
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = str(args.stanza_dir) + os.pathsep + env.get('PYTHONPATH', '')
    manifest = {'config': config, 'source_hashes': source_hashes, 'data': data, 'runtime': runtime, 'commands': dict(previous['commands'], **plan) if previous else plan,
                'pos_condition': 'gold' if args.gold_pos else 'predicted',
                'pos_action_prediction': True, 'shift_representation': 'bare_word',
                'transition_scheme': scheme_name(args), 'variant': args.variant, 'experiment_role': SYSTEM_ROLES[args.system],
                'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
                'results': {}}
    write_json(manifest_file, manifest)
    if (args.output_dir / 'results.json').exists():
        (args.output_dir / 'results.json').unlink()
    try:
        for stage, command in plan.items():
            if stage != 'train':
                # Stanza itself refuses to replace prediction files. Remove only
                # this run's old prediction pair on explicit resume/evaluate-only.
                for suffix in ['pred.mrg', 'orig.mrg']:
                    path = args.output_dir / 'predictions' / (stage + '.' + suffix)
                    if path.exists():
                        if not (args.resume or args.evaluate_only):
                            raise ValueError('Existing predictions: ' + str(path))
                        path.unlink()
            log = run_stage(stage, command, args, env)
            if stage != 'train':
                manifest['results'][stage] = evaluation_result(log,
                    args.output_dir / 'data' / (stage + '.mrg'),
                    args.output_dir / 'predictions' / (stage + '.pred.mrg'))
                write_json(manifest_file, manifest)
        manifest['status'] = 'complete'
        manifest['completed_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(args.output_dir / 'results.json', manifest['results'])
        print('\nDev F1: %.2f | Test F1: %.2f (%s POS)' % (
            manifest['results']['dev']['f1_percent'], manifest['results']['test']['f1_percent'], manifest['pos_condition']))
    except (Exception, KeyboardInterrupt) as error:
        manifest['status'] = 'failed'
        manifest['error'] = str(error)
        raise
    finally:
        write_json(manifest_file, manifest)
    return 0


def summarize_runs(runs):
    """Aggregate independent runs; report sample SD, without pooling trees."""
    seeds = [run['config']['seed'] for run in runs]
    if seeds != list(STANDARD_SEEDS):
        raise ValueError('Summary requires completed runs for seeds 1111, 2222, 3333 in order')
    reference = {k: v for k, v in runs[0]['config'].items() if k not in ('seed', 'output_dir')}
    for run in runs:
        comparable = {k: v for k, v in run['config'].items() if k not in ('seed', 'output_dir')}
        if run['status'] != 'complete' or comparable != reference:
            raise ValueError('Cannot summarize incomplete runs or different configurations')
    summary = {'seeds': list(STANDARD_SEEDS), 'number_of_runs': len(runs),
               'standard_deviation': 'sample (ddof=1)', 'shared_config': reference,
               'experiment_role': SYSTEM_ROLES[reference['system']],
               'runs': [{'seed': run['config']['seed'], 'output_dir': run['config']['output_dir'],
                         'results': run['results']} for run in runs], 'aggregate': {}}
    for split in ('dev', 'test'):
        scores = [run['results'][split]['f1'] for run in runs]
        if len({run['results'][split]['sentences'] for run in runs}) != 1:
            raise ValueError('Evaluation sentence counts differ between seeds')
        mean = statistics.mean(scores)
        sd = statistics.stdev(scores)
        summary['aggregate'][split] = {'mean_f1': mean, 'sample_sd_f1': sd,
            'mean_f1_percent': 100 * mean, 'sample_sd_f1_percent': 100 * sd,
            'sentences_per_run': runs[0]['results'][split]['sentences']}
    return summary



def write_results_markdown(path, summary):
    """Readable companion to the complete three-seed JSON summary."""
    config = summary['shared_config']
    transformer = str(config['bert_model']).replace('`', '').replace('\n', ' ')
    pos = 'Gold POS input features/targets' if config['gold_pos'] else 'Retagged POS input features/targets'
    tuning = 'Fine-tuned' if config['finetune_bert'] else 'Frozen'
    lines = [
        '# Constituency parsing results', '',
        'Parser: %s non-binary %s. Supervision: static oracle. POS labels are predicted by %s actions.' % (
            config.get('variant', 'canonical'), SYSTEM_NAMES[config['system']],
            'REDUCE(label, 1)' if config.get('variant', 'canonical') == 'canonical' and config['system'] == 'bu' else 'NT(label)'), '',
        'Experiment role: %s.' % SYSTEM_ROLES[config['system']], '',
        'Dataset: %s (`%s`).' % (LANGUAGE_PROFILES[config['lang']]['name'], config['lang']), '',
        'Seeds: **1111, 2222, 3333**. Each seed selects its own checkpoint using dev F1.', '',
        'Transformer: `%s` (%s). %s. Character models: `%s`. Attention: `%s`.' % (
            transformer, tuning.lower(), pos, config['charlm'], config['attention']), '',
        'Scores are native CoreNLP labeled bracket F1 on the **0–100 scale**.', '',
        '| Seed | Dev F1 | Test F1 |',
        '| --- | ---: | ---: |',
    ]
    for run in summary['runs']:
        lines.append('| %d | %.2f | %.2f |' % (
            run['seed'], 100 * run['results']['dev']['f1'], 100 * run['results']['test']['f1']))
    dev = summary['aggregate']['dev']
    test = summary['aggregate']['test']
    lines += [
        '| **Mean** | **%.2f** | **%.2f** |' % (dev['mean_f1_percent'], test['mean_f1_percent']),
        '| Sample SD | %.2f | %.2f |' % (dev['sample_sd_f1_percent'], test['sample_sd_f1_percent']), '',
        'Dev: **%.2f ± %.2f**. Test: **%.2f ± %.2f**.' % (
            dev['mean_f1_percent'], dev['sample_sd_f1_percent'],
            test['mean_f1_percent'], test['sample_sd_f1_percent']), '',
        'SD is the sample standard deviation across three independent runs (ddof=1), '
        'in F1 points. Predictions are not pooled or ensembled.', '',
        'Evaluated sentences per run: dev **%d**, test **%d**.' % (
            dev['sentences_per_run'], test['sentences_per_run']), '',
        'Unrounded values and the common configuration are saved in [summary.json](summary.json).', '',
        'Per-seed configurations, checkpoints, logs, and predictions:', '',
    ]
    for seed in summary['seeds']:
        lines.append('- Seed %d: [configuration](seed_%d/run.json), '
                     '[dev log](seed_%d/logs/dev.log), [test log](seed_%d/logs/test.log).' % (
                         seed, seed, seed, seed))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_series(args):
    """Use one configuration for each of the three standard seeds."""
    children = []
    for seed in STANDARD_SEEDS:
        child = copy(args)
        child.seed = seed
        child.output_dir = args.output_dir / ('seed_%d' % seed)
        # Continue remaining seeds after an interrupted batch. Completed runs
        # are re-evaluated without training; their best checkpoint is reused.
        if args.resume:
            manifest = child.output_dir / 'run.json'
            if not manifest.exists():
                child.resume = False
            elif json.loads(manifest.read_text()).get('status') == 'complete':
                child.resume = False
                child.evaluate_only = True
        children.append(child)
    if args.dry_run:
        for child in children:
            print('\n=== Seed %d | %s ===' % (child.seed, child.output_dir))
            run_single(child)
        return 0
    if args.output_dir.exists():
        allowed = {'seed_%d' % seed for seed in STANDARD_SEEDS} | {'batch.json', 'summary.json', 'results.md'}
        if args.install_corenlp:
            allowed.add('corenlp')
        if any(p.name not in allowed for p in args.output_dir.iterdir()):
            raise ValueError('Batch output directory contains unrelated files; choose a fresh directory')
        if not (args.resume or args.evaluate_only) and any(
                p.name != 'corenlp' for p in args.output_dir.iterdir()):
            raise ValueError('Batch output directory is not empty; use --resume or a fresh directory')
    # Fail before training any seed if a later seed lacks the requested model.
    for child in children:
        if child.evaluate_only and not (child.output_dir / 'models/best.pt').is_file():
            raise FileNotFoundError(str(child.output_dir / 'models/best.pt'))
        if child.resume and not (child.output_dir / 'models/checkpoint.pt').is_file():
            raise FileNotFoundError(str(child.output_dir / 'models/checkpoint.pt'))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = args.output_dir / 'summary.json'
    report_file = args.output_dir / 'results.md'
    for path in (summary_file, report_file):
        if path.exists():
            path.unlink()
    batch = {'seeds': list(STANDARD_SEEDS), 'status': 'running', 'completed_seeds': [],
             'started_utc': datetime.now(timezone.utc).isoformat()}
    write_json(args.output_dir / 'batch.json', batch)
    runs = []
    try:
        for child in children:
            print('\n=== Seed %d | %s ===' % (child.seed, child.output_dir), flush=True)
            batch['current_seed'] = child.seed
            write_json(args.output_dir / 'batch.json', batch)
            run_single(child)
            runs.append(json.loads((child.output_dir / 'run.json').read_text()))
            batch['completed_seeds'].append(child.seed)
            write_json(args.output_dir / 'batch.json', batch)
        summary = summarize_runs(runs)
        write_json(summary_file, summary)
        write_results_markdown(report_file, summary)
        batch['status'] = 'complete'
        batch['completed_utc'] = datetime.now(timezone.utc).isoformat()
        for split, metrics in summary['aggregate'].items():
            print('%s F1 over seeds 1111, 2222, 3333: %.2f +/- %.2f (sample SD)' % (
                split.capitalize(), metrics['mean_f1_percent'], metrics['sample_sd_f1_percent']))
        print('Summary: ' + str(summary_file))
        print('Results report: ' + str(report_file))
    except (Exception, KeyboardInterrupt) as error:
        batch['status'] = 'failed'
        batch['error'] = str(error)
        raise
    finally:
        write_json(args.output_dir / 'batch.json', batch)
    return 0


def main(argv=None, language='en', system='td', variant='canonical'):
    args = build_parser(language, system, variant).parse_args(argv)
    validate_options(args)
    source_files(args)
    if args.seed is not None:
        return run_single(args)
    return run_series(args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as error:
        print('Error: ' + str(error), file=sys.stderr)
        sys.exit(1)
