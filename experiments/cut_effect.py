"""What one switchable cut (a `Cuts` field) uniquely removes from a field's candidates.

The filter runs twice on the same prepared table: with the default cuts, and with one cut changed (off by default).
Objects that are candidates only in the second run are what that cut removes and nothing else would, knock-on
effects through the catalog sorting included. Optionally the |sigma| checks use approximate 90% recovery limits
(`common.M90`). Writes, under Data/fields/<field>/review/experiments/<cut>[_m90]/: summary.csv (per catalog),
removed_<catalog>.csv (every removed object with its key columns) and stamps_<catalog>.png (ZTF | PS1 for up to
--n random removed objects, in the band with the largest |sigma|).

  python experiments/cut_effect.py 860 neighbour_kron
  python experiments/cut_effect.py 860 ztf_chi_max --m90 --ps1-deteff Data/fields/000860/review/experiments/ps1_stack_deteff_860.parquet
  python experiments/cut_effect.py 860 isolation_arcsec --value 2.0 --catalogs in_ztf
"""
import argparse
import ast
import os
import sys
import warnings
from dataclasses import fields

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

from experiments.common import KEY_COLS, M90, Stamps, candidates, experiments_dir, prepare, stamp_figure
from Source_Analysis import filter_field as ff


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('field', type=int)
    p.add_argument('cut', choices=[f.name for f in fields(ff.Cuts)], help='the Cuts field to change')
    p.add_argument('--value', default=None, help='its value in the second run (a Python literal; default: off). For a cut '
                   'that is off by default, the value to test it at')
    p.add_argument('--catalogs', nargs='+', default=list(ff.CATALOG.values()))
    p.add_argument('--m90', action='store_true', help='redo the |sigma| checks with 90%% recovery limits')
    p.add_argument('--ps1-deteff', help='StackDetEffMeta table with m90 per stack (needed with --m90)')
    p.add_argument('--n', type=int, default=15, help='objects per stamp figure')
    a = p.parse_args()

    default = ff.Cuts()
    off = ast.literal_eval(a.value) if a.value is not None else (False if isinstance(getattr(default, a.cut), bool) else None)
    changed = default.replace(**{a.cut: off})
    if getattr(default, a.cut) in (None, False):                  # a cut that is off by default: --value is the cut to test
        default, changed = changed, default
        off = getattr(changed, a.cut)
    m90 = M90(a.field, a.ps1_deteff) if a.m90 else None
    out = os.path.join(experiments_dir(a.field), a.cut + ('_m90' if a.m90 else ''))
    os.makedirs(out, exist_ok=True)

    base = prepare(a.field)
    src_on, with_cut = candidates(base, default, m90)
    src_off, without = candidates(base, changed, m90)
    stamps = Stamps(a.field, src_off)
    summary = []
    for name in a.catalogs:
        removed = np.setdiff1d(without[name], with_cut[name])
        added = np.setdiff1d(with_cut[name], without[name])          # a cut can also let objects in (e.g. through the limits)
        summary.append({'catalog': name, f'candidates, {a.cut}={getattr(default, a.cut)}': len(with_cut[name]),
                        f'candidates, {a.cut}={off}': len(without[name]), 'removed only by the cut': len(removed),
                        'candidates only with the cut': len(added)})
        rows = src_off.iloc[removed]
        rows[[c for c in KEY_COLS if c in rows]].to_csv(os.path.join(out, f'removed_{name}.csv'))
        stamp_figure(stamps, src_off, removed, os.path.join(out, f'stamps_{name}.png'), f'{name}: removed only by {a.cut}', n=a.n)
    t = pd.DataFrame(summary).set_index('catalog')
    t.to_csv(os.path.join(out, 'summary.csv'))
    print(t.to_string())
    print(f'written to {out}')


if __name__ == '__main__':
    main()
