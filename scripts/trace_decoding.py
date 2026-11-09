"""Collect the actions taken in the archived constituency evaluation protocol."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

from benchmark_decoding import CORPORA, ORDERS, FAMILIES, SEEDS, digest, run_directory


def collect(args):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    sys.dont_write_bytecode = True
    source = args.work / 'stanza-main-constituency'
    if args.family == 'generalised':
        source = source.with_name(source.name + '-generalised')
    sys.path.insert(0, str(source))
    import stanza
    import torch
    from stanza.models.common.utils import sort_with_indices, unsort
    from stanza.models.constituency.trainer import Trainer
    from stanza.models.constituency.tree_reader import read_treebank
    from stanza.models.constituency.utils import retag_trees

    if Path(stanza.__file__).resolve().parent.parent != source.resolve():
        raise RuntimeError(f'Unexpected Stanza source {stanza.__file__}')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    language, expected_count = CORPORA[args.corpus]
    anchor = run_directory(args.work, args.family, args.orders[0], args.corpus, args.seed)
    test = anchor / 'data/test.mrg'
    test_hash = digest(test)
    original = read_treebank(str(test))
    if len(original) != expected_count:
        raise RuntimeError(f'Unexpected test size {len(original)}')
    device = torch.device(args.device)
    tagger = stanza.Pipeline(lang=language, dir=str(args.resources), processors='tokenize,pos',
                             tokenize_pretokenized=True, package={'pos': 'default'}, device=device,
                             download_method=None, verbose=False)
    retagged = retag_trees(original, [tagger], xpos=True)
    retagged_hash = hashlib.sha256('\n'.join(map(str, retagged)).encode()).hexdigest()
    sorted_trees, original_indices = sort_with_indices(retagged, key=len, reverse=True)
    indices = {id(tree): index for index, tree in enumerate(retagged)}
    if len(indices) != len(original):
        raise RuntimeError('Tree identities are not unique')

    args.output.mkdir(parents=True, exist_ok=True)
    for order in args.orders:
        directory = run_directory(args.work, args.family, order, args.corpus, args.seed)
        label = f'{args.family}-{order}-{args.corpus}-{args.seed}'
        path = args.output / (label + '.json.gz')
        if path.exists():
            continue
        print(f'START {label}', flush=True)
        run = json.loads((directory / 'run.json').read_text())
        if (run['status'] != 'complete' or run['config']['seed'] != args.seed
                or run['config']['system'] != order or run['data']['test']['sha256'] != test_hash
                or digest(directory / 'data/test.mrg') != test_hash):
            raise RuntimeError(f'Invalid archived run {label}')
        command = run['commands']['test']
        load_args = {key: command[command.index('--' + key) + 1] for key in
                     ('wordvec_pretrain_file', 'charlm_forward_file', 'charlm_backward_file')}
        batch_size = int(command[command.index('--eval_batch_size') + 1])
        checkpoint = directory / 'models/best.pt'
        trainer = Trainer.load(str(checkpoint), args={**load_args, 'device': device},
                               foundation_cache=tagger.foundation_cache)
        model = trainer.model
        model.eval()
        if model.reverse_sentence or model.retag_method != 'xpos':
            raise RuntimeError(f'Unexpected model settings in {label}')
        history = [[] for _ in original]
        action_codes = {action: (type(action).__name__, getattr(action, 'label', ()),
                                 getattr(action, 'arity', None)) for action in model.transitions}
        action_codes[None] = ('Stuck', (), None)

        def record(states):
            output = model.predict(states)
            for state, action in zip(states, output[1]):
                history[indices[id(state.gold_tree)]].append(action_codes[action])
            return output

        parsed = model.parse_sentences_no_grad(iter(sorted_trees), model.build_batch_from_trees,
                                               batch_size, record)
        if len(parsed) != len(original):
            raise RuntimeError(f'{label}: {len(parsed)}/{len(original)} completed parses')
        parsed = unsort(parsed, original_indices)
        archive = directory / 'predictions/test.pred.mrg'
        archived = read_treebank(str(archive))
        if len(archived) != len(parsed):
            raise RuntimeError(f'{label}: incomplete archive')
        rows = []
        for idx, (result, reference, saved) in enumerate(zip(parsed, original, archived)):
            if result.gold is not retagged[idx] or len(result.predictions) != 1:
                raise RuntimeError(f'{label}: inconsistent prediction at index {idx}')
            prediction = result.predictions[0].tree
            if prediction.leaf_labels() != reference.leaf_labels():
                raise RuntimeError(f'{label}: token yield changed at index {idx}')
            rows.append({'matches_archive': prediction.replace_tags(reference) == saved,
                         'tree': str(prediction), 'actions': history[idx]})
        saved_source = {name: sha for name, sha in run['source_hashes'].items()
                        if name.startswith('stanza/') and (source / name).is_file()}
        current_source = {name: digest(source / name) for name in saved_source}
        metadata = {'family': args.family, 'order': order, 'corpus': args.corpus,
                    'seed': args.seed, 'batch_size': batch_size, 'device': str(device),
                    'accelerator': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                    'torch': torch.__version__, 'test_sha256': test_hash,
                    'retagged_sha256': retagged_hash, 'archive_sha256': digest(archive),
                    'checkpoint_sha256': digest(checkpoint), 'root_labels': sorted(model.root_labels),
                    'source_differences': {name: [sha, current_source[name]] for name, sha in
                                           saved_source.items() if sha != current_source[name]}}
        tmp = path.with_suffix('.tmp')
        with gzip.open(tmp, 'wt', compresslevel=5) as handle:
            json.dump({'metadata': metadata, 'sentences': rows}, handle, separators=(',', ':'))
        tmp.replace(path)
        print(f'DONE {label} {sum(row["matches_archive"] for row in rows)}/{len(rows)}', flush=True)
        del record, model, trainer, parsed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--resources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--family', choices=FAMILIES, required=True)
    parser.add_argument('--corpus', choices=CORPORA, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--orders', nargs='+', choices=ORDERS, default=ORDERS)
    parser.add_argument('--device', default='cuda:0')
    collect(parser.parse_args())
