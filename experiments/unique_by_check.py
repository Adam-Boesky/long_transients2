"""What each check of the filter removes on its own, for one field, with ZTF | PS1 stamps.

For every quality check (ZTF and PS1 sides), confident check and population check, the filter is rerun with only that
check forced to pass everywhere (its other uses, e.g. in the recovery-limit references, included); the objects that
become candidates are exactly what that check removes and nothing else would (unique), knock-on effects through the
catalog sorting included. Its non-unique removals are the objects that would be in the catalog with the check passing,
fail it (in a compared band, for quality checks), and are removed by some other check too. The |sigma| checks are the selection itself and are not looped over. Optionally the |sigma| checks
use 90% recovery limits (`common.M90`).

Writes Data/fields/<field>/review/experiments/unique_by_check[_m90]/: summary.csv (checks x catalogs) and, per check,
a subdirectory with unique_<catalog>.csv, others_<catalog>.csv and stamps_<catalog>.png (up to --n unique, marked ★,
then up to --n non-unique; quality checks shown in a band that fails the check, others in the largest-|sigma| band).

  python experiments/unique_by_check.py 860
  python experiments/unique_by_check.py 860 --m90 --ps1-deteff Data/fields/000860/review/experiments/ps1_stack_deteff_860.parquet
  python experiments/unique_by_check.py 860 --checks "ZTF: DAOPhot chi <= 6" "no other object within 3\""
"""
import argparse
import os
import re
import sys
import warnings
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

from experiments.common import B3, KEY_COLS, M90, Stamps, candidates, experiments_dir, prepare, stamp_figure
from Source_Analysis import filter_field as ff

CHECK_FUNCS = ('ztf_checks', 'ps1_checks', 'real_checks', 'population_checks')


@contextmanager
def passing(label: str):
    """The filter with the check called `label` passing everywhere."""
    orig = {name: getattr(ff, name) for name in CHECK_FUNCS}

    def wrap(f):
        def g(*args, **kwargs):
            return [(l, np.ones(len(ok), bool) if l == label else ok) for l, ok in f(*args, **kwargs)]
        return g
    try:
        for name, f in orig.items():
            setattr(ff, name, wrap(f))
        yield
    finally:
        for name, f in orig.items():
            setattr(ff, name, f)


def all_checks(src: pd.DataFrame, cuts: ff.Cuts):
    """[(group, label)] of every check, in pipeline order."""
    point = src.point_source.values
    out = [('quality', l) for l, _ in ff.ztf_checks(src, 'r', point, cuts) + ff.ps1_checks(src, 'r', point, cuts)]
    for k in ff.CATALOG:
        out += [('confident', l) for l, _ in ff.real_checks(src, k, cuts) if ('confident', l) not in out]
    return out + [('population', l) for l, _ in ff.population_checks(src)]


def would_be_catalog(src: pd.DataFrame) -> np.ndarray:
    """The catalog each object is sorted into, or, where the quality checks left it not comparable, the one its
    S/N >= 5 measurements alone would put it in: in_both (0) with both surveys, in_ztf (1) / in_pstarr (2) with one."""
    z = np.logical_or.reduce([ff.ztf_detected(src, b) & (src[f'{b}_ztf_snr'].values >= ff.SNR_MIN) for b in B3])
    p = np.logical_or.reduce([src[f'{b}_ps1_meas'].values & (src[f'{b}_ps1_snr'].values >= ff.SNR_MIN) for b in B3])
    pre = np.select([z & p, z, p], [0, 1, 2], -1)
    return np.where(src.catalog.values >= 0, src.catalog.values, pre)


def failing(src: pd.DataFrame, group: str, label: str, k: int, cuts: ff.Cuts):
    """(object fails the check, {row: the band to show}): for a quality check, any band with data (measured by either
    survey) whose side fails it, whether or not the other checks drop the band too; the band shown is the failing one
    with the highest S/N on the judged side (ZTF's for ZTF checks, PS1's for PS1 checks)."""
    if group != 'quality':
        checks = dict(ff.real_checks(src, k, cuts) + ff.population_checks(src))
        return (~np.asarray(checks[label], bool) if label in checks else np.zeros(len(src), bool)), {}
    point = src.point_source.values
    side = 'ps1' if 'PS1' in label.split(':')[0] else 'ztf'
    fails, best_snr, band_of = np.zeros(len(src), bool), np.full(len(src), -np.inf), {}
    for b in 'rgi':
        checks = dict(ff.ztf_checks(src, b, point, cuts) + ff.ps1_checks(src, b, point, cuts))
        if label in checks:
            f = ~np.asarray(checks[label], bool) & (ff.ztf_detected(src, b) | src[f'{b}_ps1_meas'].values)
            snr = np.nan_to_num(src[f'{b}_{side}_snr'].values, nan=-np.inf)
            better = f & (snr > best_snr)
            for i in np.flatnonzero(better):
                band_of[int(i)] = b
            best_snr[better] = snr[better]
            fails |= f
    return fails, band_of


def check_value(src: pd.DataFrame, label: str, b: str, i: int) -> str:
    """The quantity check `label` judges, for row i (band b for quality checks), as a short string."""
    r = src.iloc[i]
    num = lambda col, fmt: (format(float(r[col]), fmt) if np.isfinite(float(r[col])) else 'none')
    if label.startswith('ZTF: outside bright-star'):
        return f'd/R {r.bright_star_d_over_R:.2f}' + (' spike' if r.in_star_spike else '')
    if label.startswith('ZTF image covers'):
        return 'off the image'
    if label.startswith('ZTF: DAOPhot flags'):
        return f"flags {int(r[f'{b}_ztf_flags'])}"
    if label.startswith('ZTF: DAOPhot chi'):
        return f"χ {num(f'{b}_ztf_chi', '.1f')}"
    if 'from a masked pixel' in label:
        return f"{num(f'{b}_ztf_nan_dist', '.1f')} px"
    if label.startswith('extended: ZTF Kron'):
        return f"{num(f'{b}_ztf_kron_masked', '.2f')} masked"
    if 'polar skycell' in label:
        return 'bad skycell'
    if 'catalog covers' in label:
        return 'DR2 missing'
    if "within 1'" in label:
        return "nothing within 1'"
    if 'residual sky' in label:
        return f"sky {num(f'{b}_ps1_sky', '.1e')}"
    if 'nFrames' in label:
        return f"nFrames {num(f'{b}_ps1_nFrames', '.0f')}"
    if label.startswith('PS1: not ') and label[9:] in ('DEFECT', 'PEAK_ON_SPIKE', 'SATSTAR'):
        return f"infoFlag 0x{int(r[f'{b}_ps1_infoFlag']):x}"
    if label.startswith('PS1: not ') and label[9:] in ('ON_SPIKE', 'ON_STARCORE', 'ON_BURNTOOL', 'SATSTAR_PROFILE'):
        return f"infoFlag2 0x{int(r[f'{b}_ps1_infoFlag2']):x}"
    if 'measured in the band' in label:
        return 'no flux'
    if 'psfQfPerfect' in label:
        return f"psfQf {num(f'{b}_ps1_psfQfPerfect', '.2f')}"
    if label.startswith('no other object within'):
        return f'{r.nearest_other_arcsec:.1f}″'
    if 'Kron aperture' in label:
        return 'in a brighter Kron'
    if 'sharp' in label:
        return f'|sharp| {abs(ff.ztf_sharpness(src.iloc[[i]])[0]):.2f}'
    if 'nStackDetections' in label:
        return f'nStack {int(r.ps1_nStackDetections)}'
    if 'STACK' in label:
        return f'qualityFlag {int(r.ps1_qualityFlag)}'
    if 'proper motion' in label:
        return f'pm {ff.gaia_pm_significance(src.iloc[[i]])[0]:.1f}σ'
    if 'parallax' in label:
        return f'ϖ/σ {r.gaia_parallax_over_error:.1f}'
    if '|σ|' in label or 'same sign' in label:
        return ' '.join(f"{x}{r[f'{x}_sigma']:+.1f}" for x in B3 if r[f'{x}_cat'] >= 0 and np.isfinite(r[f'{x}_sigma'])) or 'no σ'
    return 'fails'


class Failures:
    """Which checks each row of a filter run's table fails: the quality checks per band, its catalog's object checks
    and the |sigma| checks."""

    def __init__(self, src: pd.DataFrame, cuts: ff.Cuts):
        self.src = src
        point = src.point_source.values
        self.quality = {b: dict(ff.ztf_checks(src, b, point, cuts) + ff.ps1_checks(src, b, point, cuts)) for b in B3}
        sigma = ff.interesting_checks(src) if all(f'{b}_sigma' in src for b in B3) else []
        self.objects = {k: dict(ff.real_checks(src, k, cuts) + ff.population_checks(src) + sigma) for k in ff.CATALOG}

    def caption(self, i: int, label: str, group: str) -> str:
        k = int(self.src.catalog.values[i])
        cat = {b: self.src[f'{b}_cat'].values[i] >= 0 for b in B3}
        if group == 'quality':
            this = '  '.join(f'{b} {check_value(self.src, label, b, i)}' for b in B3
                             if label in self.quality[b] and not self.quality[b][label][i] and cat[b]) or 'passes in compared bands'
        else:
            this = check_value(self.src, label, 'r', i)
        others = []
        for b in B3:
            others += [f'{l} [{b}] ({check_value(self.src, l, b, i)})' for l, ok in self.quality[b].items() if l != label and not ok[i]]
        if k in self.objects:
            others += [f'{l} ({check_value(self.src, l, "r", i)})' for l, ok in self.objects[k].items() if l != label and not ok[i]]
        tail = ''
        if len(others) > 3:
            others, tail = others[:3], f'  +{len(others) - 3} more'
        lines = [f'this check: {this}', 'also fails: ' + ('none' if not others else others[0])] + others[1:]
        return '\n'.join(lines) + tail


def slug(label: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_')[:60]


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('field', type=int)
    p.add_argument('--checks', nargs='+', help='only these check labels (default: every check)')
    p.add_argument('--m90', action='store_true', help='redo the |sigma| checks with 90%% recovery limits')
    p.add_argument('--ps1-deteff', help='StackDetEffMeta table with m90 per stack (needed with --m90)')
    p.add_argument('--n', type=int, default=10, help='objects per stamp figure')
    a = p.parse_args()

    cuts = ff.Cuts()
    m90 = M90(a.field, a.ps1_deteff) if a.m90 else None
    out = os.path.join(experiments_dir(a.field), 'unique_by_check' + ('_m90' if a.m90 else ''))
    os.makedirs(out, exist_ok=True)
    base = prepare(a.field)
    src, nominal = candidates(base, cuts, m90)
    checks = [(g, l) for g, l in all_checks(src, cuts) if not a.checks or l in a.checks]
    print('nominal candidates:', {k: len(v) for k, v in nominal.items()})
    summary = []
    for group, label in checks:
        with passing(label):
            src_off, without = candidates(base, cuts, m90)
        d = os.path.join(out, f'{group}_{slug(label)}')
        os.makedirs(d, exist_ok=True)
        stamps = Stamps(a.field, src_off)
        fails_of = Failures(src_off, cuts)
        row = {'group': group, 'check': label}
        for k, name in ff.CATALOG.items():
            unique = np.setdiff1d(without[name], nominal[name])
            fails, band_of = failing(src_off, group, label, k, cuts)
            others = np.setdiff1d(np.flatnonzero(fails & (would_be_catalog(src_off) == k)), without[name])
            row[name], row[f'{name} (with others)'] = len(unique), len(others)
            for tag, rr in (('unique', unique), ('others', others)):
                t = src_off.iloc[rr]
                t[[c for c in KEY_COLS if c in t]].to_csv(os.path.join(d, f'{tag}_{name}.csv'))
            stamp_figure(stamps, src_off, unique, os.path.join(d, f'stamps_{name}.png'), f'{name}: removed by\n{label}',
                         n=a.n, band_of=band_of, starred=unique, others=others,
                         captions=lambda i, label=label, group=group: fails_of.caption(i, label, group))
        summary.append(row)
        print(row, flush=True)
        pd.DataFrame(summary).to_csv(os.path.join(out, 'summary.csv'), index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f'written to {out}')


if __name__ == '__main__':
    main()
