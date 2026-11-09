#!/usr/bin/env python3
"""Static canonical BU on PTB, using the shared runner."""
import sys
from run_canonical_static import main

if __name__ == '__main__':
    try:
        sys.exit(main(language='en', system='bu'))
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as error:
        print('Error: ' + str(error), file=sys.stderr)
        sys.exit(1)
