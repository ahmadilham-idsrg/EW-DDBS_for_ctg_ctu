"""
check_notebook_sync.py
======================
The Colab notebooks carry their own copies of the source modules, written to
disk with `%%writefile`. That is convenient for a reader with no local setup,
and dangerous for a maintainer: editing `src/` leaves the notebooks silently
stale, and a reviewer who runs the notebook then obtains different code from
the one the repository advertises.

This script compares every embedded module against its counterpart in `src/`
and exits non-zero on any divergence.

Usage
-----
    python scripts/check_notebook_sync.py
    python scripts/check_notebook_sync.py --fix     # rewrite notebooks from src/
"""

import argparse
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), '..')


def embedded_modules(nb):
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        if not src.startswith('%%writefile'):
            continue
        first, _, body = src.partition('\n')
        yield i, first.replace('%%writefile', '').strip(), body


def main(fix):
    stale = []
    checked = 0
    for path in sorted(glob.glob(os.path.join(ROOT, 'notebooks', '*.ipynb'))):
        nb = json.load(open(path))
        changed = False
        for idx, name, body in embedded_modules(nb):
            disk_path = os.path.join(ROOT, 'src', name)
            if not os.path.exists(disk_path):
                print(f'  MISSING  {os.path.basename(path)} embeds {name}, '
                      f'which is absent from src/')
                stale.append((path, name))
                continue
            disk = open(disk_path).read()
            checked += 1
            if body.rstrip() == disk.rstrip():
                print(f'  OK       {os.path.basename(path):28s} {name}')
                continue
            print(f'  STALE    {os.path.basename(path):28s} {name}')
            stale.append((path, name))
            if fix:
                text = f'%%writefile {name}\n' + disk.rstrip('\n')
                nb['cells'][idx]['source'] = [l + '\n' for l in text.split('\n')]
                nb['cells'][idx]['source'][-1] = \
                    nb['cells'][idx]['source'][-1].rstrip('\n')
                changed = True
        if changed:
            json.dump(nb, open(path, 'w'), indent=1, ensure_ascii=False)
            print(f'  FIXED    {os.path.basename(path)} rewritten from src/')

    print(f'\n{checked} embedded modules checked, {len(stale)} stale')
    if stale and not fix:
        print('Run with --fix to rewrite the notebooks from src/.')
        return 1
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fix', action='store_true')
    sys.exit(main(ap.parse_args().fix))
