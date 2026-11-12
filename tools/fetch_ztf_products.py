"""Download ZTF deep reference products for the whole footprint, one keep-alive stream per client IP.

IRSA allows one transfer at a time per client IP (a second connection gets an instant 503) and
feeds Cannon's at ~0.5 MB/s, so the only lever is more egress IPs: Cannon NATs each rack family
(holy7c, holy8a, ...) through its own address. Workers therefore cooperate through the shared
filesystem rather than fixed shards: each claims one *field* at a time by creating
`<data_dir>/fetch_logs/claims_<product>/field<6>/` (heartbeat-touched; a claim idle for
STALE_MINUTES is taken over), and with `--one-per-ip` a worker whose egress IP is already served by a
live worker exits at once, so a job array can be oversubscribed across partitions safely.

Progress lives on disk: a product is done when its file exists under `<data_dir>/field<6>/`, a 404 is
recorded in `fetch_logs/missing_<product>.txt`, and a rerun resumes wherever it stopped.

    python tools/fetch_ztf_products.py --products refpsfcat --claim --one-per-ip   # a cooperating worker
    python tools/fetch_ztf_products.py --products refpsfcat                        # a lone stream, fixed order
    python tools/fetch_ztf_products.py --products refpsfcat --status               # count done / missing
"""
import argparse
import os
import socket
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Extracting.utils import get_data_path  # noqa: E402
from Extracting.ztf_images import fetch_product, field_dir, refimg_path, refimg_url  # noqa: E402

MANIFEST = Path(__file__).resolve().parents[1] / 'Extracting' / 'footprint' / 'irsa_deep_quadrants.csv'
STALE_MINUTES = 20.0          # a claim or IP lease not touched for this long belongs to a dead worker
WORKER = f"{os.environ.get('SLURM_JOB_ID', 'nojob')}@{socket.gethostname()}"


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


def _lease(path: str, owner: str = WORKER) -> bool:
    """Atomically take `path` as a lease; a stale one (no heartbeat for STALE_MINUTES) is taken over."""
    try:
        os.mkdir(path)
    except FileExistsError:
        try:
            age_min = (time.time() - os.stat(path).st_mtime) / 60.0
        except FileNotFoundError:
            return _lease(path, owner)
        if age_min < STALE_MINUTES:
            return False
        print(f'FETCH_TAKEOVER stale lease {path} ({age_min:.0f} min)', flush=True)
    with open(os.path.join(path, 'owner'), 'w') as f:
        f.write(owner + '\n')
    os.utime(path)
    return True


def _heartbeat(path: str) -> None:
    try:
        os.utime(path)
    except FileNotFoundError:
        pass


def _release(path: str) -> None:
    try:
        os.remove(os.path.join(path, 'owner'))
        os.rmdir(path)
    except OSError:
        pass


def public_ip() -> str:
    """Egress address as IRSA sees it; empty if it cannot be found (then no IP lease is enforced)."""
    for url in ('https://api.ipify.org', 'https://ifconfig.me/ip'):
        try:
            r = requests.get(url, timeout=10)
            if r.ok and r.text.strip():
                return r.text.strip()
        except requests.exceptions.RequestException:
            pass
    return ''


def status(manifest: pd.DataFrame, data_dir: str, products) -> None:
    for product in products:
        missing = read_missing(data_dir, product)
        done = sum(os.path.exists(refimg_path(field_dir(data_dir, r.field), r.field, r.ccdid, r.qid, r.band, product))
                   for r in manifest.itertuples())
        n404 = sum(os.path.basename(refimg_path('', r.field, r.ccdid, r.qid, r.band, product)) in missing
                   for r in manifest.itertuples())
        print(f'{product}: {done}/{len(manifest)} on disk, {n404} missing on IRSA, {len(manifest) - done - n404} to go')
    for kind in ('ip',) + tuple(f'claims_{p}' for p in products):
        root = os.path.join(data_dir, 'fetch_logs', kind)
        if os.path.isdir(root):
            for d in sorted(os.listdir(root)):
                path = os.path.join(root, d)
                owner = open(os.path.join(path, 'owner')).read().strip() if os.path.exists(os.path.join(path, 'owner')) else '?'
                age = (time.time() - os.stat(path).st_mtime) / 60.0
                print(f'  {kind}/{d}: {owner}, heartbeat {age:.1f} min ago{" (STALE)" if age > STALE_MINUTES else ""}')


def fetch_rows(rows: pd.DataFrame, data_dir: str, product: str, session: requests.Session, counts: dict,
               t0: float, missing: set, miss, heartbeat: str = None) -> None:
    for r in rows.itertuples():
        dest = refimg_path(field_dir(data_dir, r.field), r.field, r.ccdid, r.qid, r.band, product)
        name = os.path.basename(dest)
        if name in missing:
            counts['404'] += 1
            continue
        result = fetch_product(session, refimg_url(r.field, r.ccdid, r.qid, r.band, product), dest)
        counts[result] += 1
        if heartbeat:
            _heartbeat(heartbeat)
        if result == '404':
            miss.write(name + '\n')
            miss.flush()
        if result in ('ok', '404', 'fail'):
            rate = (counts['ok'] + counts['404']) / (time.time() - t0)
            print(f'FETCH_{result.upper()} {name} {counts} {rate:.2f} files/s', flush=True)


def run(manifest: pd.DataFrame, data_dir: str, products, shard: str, reverse: bool = False,
        claim: bool = False, one_per_ip: bool = False) -> None:
    i, n = (int(v) for v in shard.split('/'))
    manifest = manifest.iloc[i::n]
    if reverse:
        manifest = manifest.iloc[::-1]
    logs = os.path.join(data_dir, 'fetch_logs')
    os.makedirs(logs, exist_ok=True)
    ip_lease = None
    if one_per_ip:
        ip = public_ip()
        if ip:
            ip_lease = os.path.join(logs, 'ip', ip)
            os.makedirs(os.path.dirname(ip_lease), exist_ok=True)
            if not _lease(ip_lease):
                print(f'FETCH_EXIT egress IP {ip} already served by {open(os.path.join(ip_lease, "owner")).read().strip()}', flush=True)
                return
            print(f'FETCH_START {WORKER} on egress IP {ip}', flush=True)
        else:
            print('FETCH_START could not determine the egress IP; running without an IP lease', flush=True)
    session = requests.Session()
    counts = {'ok': 0, 'cached': 0, '404': 0, 'fail': 0}
    t0 = time.time()
    for product in products:
        missing = read_missing(data_dir, product)
        with open(missing_log(data_dir, product), 'a') as miss:
            if not claim:
                fetch_rows(manifest, data_dir, product, session, counts, t0, missing, miss, heartbeat=ip_lease)
                continue
            claims = os.path.join(logs, f'claims_{product}')
            os.makedirs(claims, exist_ok=True)
            for field, rows in manifest.groupby('field', sort=False):
                if all(os.path.exists(refimg_path(field_dir(data_dir, r.field), r.field, r.ccdid, r.qid, r.band, product))
                       or os.path.basename(refimg_path('', r.field, r.ccdid, r.qid, r.band, product)) in missing
                       for r in rows.itertuples()):
                    continue
                lease = os.path.join(claims, f'field{int(field):06d}')
                if not _lease(lease):
                    continue
                print(f'FETCH_CLAIM field {field} ({len(rows)} files) by {WORKER}', flush=True)
                fetch_rows(rows, data_dir, product, session, counts, t0, missing, miss, heartbeat=lease)
                if ip_lease:
                    _heartbeat(ip_lease)
                _release(lease)
    if ip_lease:
        _release(ip_lease)
    print(f'FETCH_DONE {counts} in {(time.time() - t0) / 3600:.1f} h', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--products', nargs='+', default=['refpsfcat'])
    ap.add_argument('--shard', default='0/1', help='i/n: this process takes every n-th file starting at i')
    ap.add_argument('--fields', nargs='*', type=int, help='restrict to these field IDs')
    ap.add_argument('--reverse', action='store_true', help='walk the manifest from the last field backwards')
    ap.add_argument('--claim', action='store_true', help='cooperate with other workers by claiming one field at a time')
    ap.add_argument('--one-per-ip', action='store_true', help='exit if another live worker shares this egress IP')
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
        run(manifest, args.data_dir, args.products, args.shard, args.reverse, args.claim, args.one_per_ip)


if __name__ == '__main__':
    main()
