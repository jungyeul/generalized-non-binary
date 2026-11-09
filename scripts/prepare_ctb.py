"""Convert licensed CTB 5.1 bracketed files into the paper's basic split."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bracketed', type=Path, help='Directory containing CTB bracketed files')
    parser.add_argument('output', type=Path, help='New directory for zh_ctb_{train,dev,test}.mrg')
    args = parser.parse_args()
    if not args.bracketed.is_dir() or not any(args.bracketed.iterdir()):
        parser.error('Provide the nonempty bracketed directory from the licensed CTB 5.1 release')
    if args.output.exists():
        parser.error('Use a new output directory')
    source = Path(__file__).resolve().parents[1] / 'stanza-main-constituency-generalised'
    sys.path.insert(0, str(source))
    from stanza.utils.datasets.constituency.convert_ctb import Version, convert_ctb

    args.output.mkdir(parents=True)
    convert_ctb(str(args.bracketed), str(args.output), 'zh_ctb', Version.V51b)


if __name__ == '__main__':
    main()
