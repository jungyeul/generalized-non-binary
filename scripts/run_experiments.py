"""Launch the paper's six Stanza parsers on PTB and CTB."""
import argparse
from pathlib import Path
import subprocess
import sys

from benchmark_decoding import CORPORA, FAMILIES, ORDERS, run_directory

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = {'ptb': ('PTB_3', 'en_ptb'), 'ctb': ('CTB_5.1', 'zh_ctb')}
SPLITS = ('train', 'dev', 'test')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--ptb-dir', type=Path, help='Licensed WSJ directory with sections 02 through 23')
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=FAMILIES)
    parser.add_argument('--corpora', nargs='+', choices=CORPORA, default=tuple(CORPORA))
    parser.add_argument('--orders', nargs='+', choices=ORDERS, default=ORDERS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resources-dir', type=Path)
    core = parser.add_mutually_exclusive_group()
    core.add_argument('--corenlp-home', type=Path)
    core.add_argument('--install-corenlp', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    corpora = {}
    for corpus in args.corpora:
        if corpus == 'ptb' and args.ptb_dir is not None:
            source = args.ptb_dir.expanduser().resolve()
            for section in range(2, 24):
                if not any((source / f'{section:02}').rglob('*.mrg')):
                    parser.error(f'Missing .mrg files for WSJ section {section:02} in {source}')
            corpora[corpus] = ['--ptb-dir', str(source)]
            continue
        directory, prefix = LAYOUT[corpus]
        splits = {split: args.data_root.expanduser().resolve() / directory / f'{prefix}_{split}.mrg'
                  for split in SPLITS}
        data_args = []
        for split, path in splits.items():
            if not path.is_file():
                parser.error(f'Missing treebank input {path}')
            data_args.extend(('--' + split, str(path)))
        corpora[corpus] = data_args

    common = ['--device', args.device]
    for name in ('resources_dir', 'corenlp_home'):
        path = getattr(args, name)
        if path is not None:
            common.extend(('--' + name.replace('_', '-'), str(path.expanduser().resolve())))
    for name in ('install_corenlp', 'resume', 'evaluate_only', 'dry_run'):
        if getattr(args, name):
            common.append('--' + name.replace('_', '-'))

    for family in dict.fromkeys(args.families):
        checkout = ROOT / ('stanza-main-constituency' +
                           ('-generalised' if family == 'generalised' else ''))
        for corpus, data_args in corpora.items():
            for order in dict.fromkeys(args.orders):
                output = run_directory(ROOT, family, order, corpus, 1111).parent
                runner = checkout / 'scripts' / f'run_{family}_{order}_{corpus}.py'
                command = [sys.executable, str(runner), '--output-dir', str(output),
                           *common, *data_args]
                print(f'{family} {order} on {corpus.upper()}', flush=True)
                subprocess.run(command, cwd=checkout, check=True)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
