#!/usr/bin/env python3
"""Future-reference IO: PTB seeds/summary, then CTB seeds/summary."""
import signal
import sys
from run_canonical_experiments import main, interrupted

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main(system='io'))
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as error:
        print('Error: ' + str(error), file=sys.stderr)
        sys.exit(1)
