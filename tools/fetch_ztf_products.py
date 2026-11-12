"""Download ZTF deep reference products for the whole footprint, one keep-alive stream per process.

IRSA throttles each client to roughly one file every two seconds and answers 503 to anything faster,
so the download is latency-bound and the only levers are a persistent connection and more clients on
different hosts (`--shard i/n`). Progress lives on disk: a product is done when its file exists under
`<data_dir>/field<6>/` and a 404 is recorded in `<data_dir>/fetch_logs/missing_<product>.txt`, so a
rerun resumes wherever it stopped.

    python tools/fetch_ztf_products.py --products refpsfcat                 # everything, one stream
    python tools/fetch_ztf_products.py --products refpsfcat --shard 1/2     # second of two hosts
    python tools/fetch_ztf_products.py --products refpsfcat --status        # count done / missing
"""
import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Extracting.utils import get_data_path  # noqa: E402
from Extracting.ztf_images import fetch_product, field_dir, refimg_path, refimg_url  # noqa: E402

MANIFEST = Path(__file__).resolve().parents[1] / 'Extracting' / 'footprint' / 'irsa_deep_quadrants.csv'


def load_manifest(fields=None) -> pd.DataFrame:
    """IRSA's deep-reference listing (field, ccdid, qid, filtercode), ordered so fields complete together."""
    m = pd.read_csv(MANIFEST)
    m['band'] = m['filtercode'].str[1]
    if fields:
        m = m[m['field'].isin(fields)]
    return m.sort_values(['field', 'ccdid', 'qid', 'band']).reset_index(drop=True)


def missing_log(data_dir: str, product: str) -> str:
    return os.path.join(data_dir, 'fetch_logs', f'missing_{product}.txt')


def read_missing(data_dir: str, product: str) -> set:
    path = missing_log(data_dir, product)
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def status(manifest: pd.DataFrame, data_dir: str, products) -> None:
    for product in products:
        missing = read_missing(data_dir, product)
        done = sum(os.path.exists(refimg_path(field_dir(data_dir, r.field), r.field, r.ccdid, r.qid, r.band, product))
                   for r in manifest.itertuples())
        n404 = sum(os.path.basename(refimg_path('', r.field, r.ccdid, r.qid, r.band, product)) in missing
                   for r in manifest.itertuples())
        print(f'{product}: {done}/{len(manifest)} on disk, {n404} missing on IRSA, {len(manifest) - done - n404} to go')


def run(manifest: pd.DataFrame, data_dir: str, products, shard: str) -> None:
    i, n = (int(v) for v in shard.split('/'))
    manifest = manifest.iloc[i::n]
    os.makedirs(os.path.join(data_dir, 'fetch_logs'), exist_ok=True)
    session = requests.Session()
    counts = {'ok': 0, 'cached': 0, '404': 0, 'fail': 0}
    t0 = time.time()
    for product in products:
        missing = read_missing(data_dir, product)
        with open(missing_log(data_dir, product), 'a') as miss:
            for k, r in enumerate(manifest.itertuples(), 1):
                dest = refimg_path(field_dir(data_dir, r.field), r.field, r.ccdid, r.qid, r.band, product)
                name = os.path.basename(dest)
                if name in missing:
                    counts['404'] += 1
                    continue
                result = fetch_product(session, refimg_url(r.field, r.ccdid, r.qid, r.band, product), dest)
                counts[result] += 1
                if result == '404':
                    miss.write(name + '\n')
                    miss.flush()
                if result in ('ok', '404', 'fail'):
                    rate = (counts['ok'] + counts['404']) / (time.time() - t0)
                    print(f'FETCH_{result.upper()} {name} [{k}/{len(manifest)}] {counts} {rate:.2f} files/s', flush=True)
    print(f'FETCH_DONE {counts} in {(time.time() - t0) / 3600:.1f} h', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--products', nargs='+', default=['refpsfcat'])
    ap.add_argument('--shard', default='0/1', help='i/n: this process takes every n-th file starting at i')
    ap.add_argument('--fields', nargs='*', type=int, help='restrict to these field IDs')
    ap.add_argument('--limit', type=int, help='stop after this many manifest rows (smoke tests)')
    ap.add_argument('--data-dir', default=os.path.join(get_data_path(), 'ztf_data'))
    ap.add_argument('--status', action='store_true', help='report progress and exit')
    args = ap.parse_args()
    manifest = load_manifest(args.fields)
    if args.limit:
        manifest = manifest.head(args.limit)
    if args.status:
        status(manifest, args.data_dir, args.products)
    else:
        run(manifest, args.data_dir, args.products, args.shard)


if __name__ == '__main__':
    main()
