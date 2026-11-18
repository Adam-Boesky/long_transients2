import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import astropy.units as u
from pathlib import Path

from typing import List, Iterable, Tuple, Optional
from ztfquery import lightcurve
from astropy.time import Time
from astropy.table import Table, vstack
from astroquery.sdss import SDSS
from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
from concurrent.futures import ThreadPoolExecutor

from Extracting.utils import img_flux_to_ab_mag
from Source_Analysis.casjobs import get_pstarr_lc_from_coord, get_pstarr_lc_from_id
from Source_Analysis.ZTF_Local_LC import ZTF_LC, BAD_CATFLAG_MASK
from Source_Analysis import followup

# Diagnostic: log SDSS's raw response body when it fails to parse, so we can see
# *why* (rate limit, server error, etc.) instead of just the resulting parse exception.
import astroquery.sdss.core as _sdss_core
_orig_sdss_parse_result = _sdss_core.SDSSClass._parse_result



# Per-process SDSS rate-limit lock (set by init_worker_sdss_lock in parallel contexts).
# SDSS SkyServer enforces a hard cap of 60 queries/minute (>=1 second between queries) and
# blocks with an HTTP 403 if that's exceeded:
# https://skyserver.sdss.org/dr14/en/help/docs/limits.aspx
SDSS_MIN_QUERY_INTERVAL_SEC = 1.1
_sdss_lock = None
_sdss_last_call_time = None


def init_worker_sdss_lock(lock, last_call_time) -> None:
    """Initializer for Pool/ProcessPoolExecutor workers — stores the shared SDSS rate-limit lock."""
    global _sdss_lock, _sdss_last_call_time
    _sdss_lock = lock
    _sdss_last_call_time = last_call_time


def throttled_sdss_query(query_fn, *args, **kwargs):
    """Call query_fn (an SDSS query), pausing as needed so calls across all worker processes
    stay under SDSS's rate limit."""
    if _sdss_lock is None:
        return query_fn(*args, **kwargs)
    with _sdss_lock:
        wait = SDSS_MIN_QUERY_INTERVAL_SEC - (time.time() - _sdss_last_call_time.value)
        if wait > 0:
            time.sleep(wait)
        try:
            return query_fn(*args, **kwargs)
        finally:
            _sdss_last_call_time.value = time.time()


def _sdss_parse_result_with_diagnostics(self, response, verbose=False):
    try:
        return _orig_sdss_parse_result(self, response, verbose=verbose)
    except Exception:
        print(
            f'SDSS response failed to parse (status={response.status_code}). '
            f'Raw body (first 1000 chars): {response.text[:1000]!r}'
        )
        raise


_sdss_core.SDSSClass._parse_result = _sdss_parse_result_with_diagnostics

ALL_LC_COLNAMES = ['ptf_id', 'wise_id', 'ztf_id', 'ra', 'dec', 'mjd', 'g_mag', 'g_magerr', 'r_mag', 'r_magerr', 'i_mag',
                   'i_magerr', 'w1_mag', 'w1_magerr', 'w2_mag', 'w2_magerr', 'w3_mag', 'w3_magerr', 'w4_mag',
                   'w4_magerr', 'R_mag', 'R_magerr']
def wise_colors(lc: Table, color_err_thresh: float = 0.5):
    """W1 - W2 per epoch with sigma(W1-W2) = sqrt(sigma_W1^2 + sigma_W2^2) below
    `color_err_thresh`, or None when the light curve has no W1/W2 errors.

    The one definition behind the analysis page's W1-W2 histogram and summary
    and the website's `wise_w1w2_mean` / `wise_w1w2_median` filter columns.
    """
    if 'w1_magerr' not in lc.colnames or 'w2_magerr' not in lc.colnames:
        return None
    w1_err = lc['w1_magerr'].filled(fill_value=np.nan)
    w2_err = lc['w2_magerr'].filled(fill_value=np.nan)
    color_mask = np.sqrt(w1_err**2 + w2_err**2) < color_err_thresh
    return (lc['w1_mag'] - lc['w2_mag'])[color_mask]


LC_MARKER_INFO = {
    'ptf': 'P',
    'ztf': '<',
    'sdss': 'd',
    'panstarrs': '3',
    'gaia': 'X',
    'custom': '.',
    'ztforce': 's',
    'zubercal': 'D',
}
LC_COLOR_INFO = {
    'u': 'black',
    'g': 'forestgreen',
    'r': 'firebrick',
    'i': 'darkorchid',
    'z': 'lightcoral',
    'y': 'gold',
    'b': 'steelblue',
}
ALL_BAND_DF = pd.DataFrame(
    data={
        'g_mag': ['g', 'ztf'],
        'r_mag': ['r', 'ztf'],
        'i_mag': ['i', 'ztf'],
        'G_mag': ['g', 'ptf'],
        'R_mag': ['r', 'ptf'],
        'uModel_mag': ['u', 'sdss'],
        'gModel_mag': ['g', 'sdss'],
        'rModel_mag': ['r', 'sdss'],
        'iModel_mag': ['i', 'sdss'],
        'zModel_mag': ['z', 'sdss'],
        'yModel_mag': ['y', 'sdss'],
        'pstarr_g_mag': ['g', 'panstarrs'],
        'pstarr_r_mag': ['r', 'panstarrs'],
        'pstarr_i_mag': ['i', 'panstarrs'],
        'pstarr_z_mag': ['z', 'panstarrs'],
        'pstarr_y_mag': ['y', 'panstarrs'],
        'gaia_g_mag': ['g', 'gaia'],
        'gaia_rp_mag': ['r', 'gaia'],
        'gaia_bp_mag': ['b', 'gaia'],
        'custom_g_mag': ['g', 'custom'],
        'custom_r_mag': ['r', 'custom'],
        'custom_i_mag': ['i', 'custom'],
        'custom_z_mag': ['z', 'custom'],
        'ztforce_g_mag': ['g', 'ztforce'],
        'ztforce_r_mag': ['r', 'ztforce'],
        'ztforce_i_mag': ['i', 'ztforce'],
        'zubercal_g_mag': ['g', 'zubercal'],
        'zubercal_r_mag': ['r', 'zubercal'],
        'zubercal_i_mag': ['i', 'zubercal'],
    },
    index=['band', 'survey'],
)
GAIA_ZERO_PTS = {'g': 25.8010, 'bp': 25.3540, 'rp': 25.1040}  # from Table 5.4 in https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photProc/cu5pho_ssec_photCal.html#SSS3.P2


class Light_Curve:
    def __init__(
            self,
            ra: float,
            dec: float,
            catalogs: List[str] = ['ztf', 'wise', 'neowise', 'ptf', 'sdss', 'panstarrs', 'gaia', 'custom'],
            query_rad_arcsec: float = 1.5,
            query_in_parallel: bool = True,
            pstarr_objid: Optional[int] = None,
            pstarr_coord: Optional[Tuple[float]] = None,
            ztf_local_dir: Optional[str] = None,
            store=None,
        ):
        self.ra = ra
        self.dec = dec
        self.skycoord = SkyCoord(ra, dec, unit="deg")
        self.catalogs = catalogs
        self.query_rad_arcsec = query_rad_arcsec  # query radius in arcseconds
        self.query_in_parallel = query_in_parallel  # whether we should query the apis in parallel or not
        self.pstarr_objid = pstarr_objid
        self.pstarr_coord = pstarr_coord
        self.ztf_local_dir = ztf_local_dir  # if set, load ZTF LCs from local parquet files instead of the API
        # A `local_store.LocalStore`, or None for live queries. Consulted
        # in `get_catalog_lc` *before* the network, and only for fetching:
        # the shaped table it returns still goes through
        # `_finish_catalog_lc`, so the binning is the same code either way.
        self.store = store
        self.source_uuid = None  # set by Source when it knows one

        # Map class catalog names to astroquery catalog names
        self.catalog_astroquery_map = {
            'ztf': 'ztf_objects_dr24',
            'wise': 'allwise_p3as_mep',
            'neowise': 'neowiser_p1bs_psd',
            'ptf': 'ptf_lightcurves',
            'gaia': 'gaia_dr3_source',
        }

        self._lc = None

    @property
    def lc(self) -> Table:
        if self._lc is None:
            self._lc = self.get_lc()
        return self._lc

    @staticmethod
    def _cluster_by_gap(sorted_mjd: np.ndarray, max_gap: float = 5.0) -> np.ndarray:
        """Group already-time-sorted mjds into clusters (e.g. WISE/NEOWISE single-visit
        exposures) by starting a new cluster wherever the gap to the *previous* point
        exceeds `max_gap`, rather than measuring a fixed window from the cluster's first
        point. The latter can split one real visit into pieces if its total span happens
        to exceed `max_gap` even though every internal gap between exposures is small.

        Returns an integer cluster id per row (0, 0, 1, 1, 1, 2, ...).
        """
        if len(sorted_mjd) == 0:
            return np.array([], dtype=int)
        starts_new_cluster = np.concatenate(([True], np.diff(sorted_mjd) > max_gap))
        return np.cumsum(starts_new_cluster) - 1

    @staticmethod
    def _bin_visit_clusters(lightcurve_tab, bands, max_gap: float = 5.0):
        """Average same-visit exposures into one row per visit, per band.

        Used for both WISE and NEOWISE. They are combined by the same rule because
        they measure the same thing the same way; the only structural difference is
        that WISE carries w1-w4 and no per-exposure SNR column, while NEOWISE carries
        w1-w2 and does.

        Magnitudes are combined by inverse-variance weighting in *flux*, matching
        AllWISE's own WPRO methodology (AllWISE Explanatory Supplement sec2_3b /
        sec4_4c): averaged single-exposure magnitudes are computed from the
        inverse-variance-weighted profile-fit flux measurements, not by averaging
        magnitudes. The reported error is the error *on the mean*, not the typical
        per-exposure error -- taking `np.nanmean` of the magerr column (as the WISE
        branch used to) reports roughly the single-exposure precision, which does not
        shrink with the ~11 exposures in a visit and overstates the uncertainty by
        about sqrt(N) (measured: a factor of 3.4 across 1,982 WISE visits).

        Against a *correctly* computed plain-mean error, sqrt(sum(sigma_i^2))/N,
        inverse-variance weighting is only ~5% better here -- the exposures within a
        visit have similar uncertainties, and the two estimators coincide exactly when
        they are equal. It is preferred for being the minimum-variance combination and
        for matching how the AllWISE catalogue magnitudes themselves are built, not
        for the size of the number.

        Upper limits (null magerr, SNR<2; NEOWISE Explanatory Supplement sec2_1c) carry
        no per-point uncertainty, so real detections are preferred within a cluster and
        upper limits are only averaged in when a cluster has no detection in that band.
        """
        if len(lightcurve_tab) == 0:
            return lightcurve_tab

        lightcurve_tab.sort('mjd')
        band_cols = [f'{band}_{suffix}' for band in bands
                     for suffix in ('mag', 'magerr', 'snr')
                     if f'{band}_{suffix}' in lightcurve_tab.colnames]
        clusters = Light_Curve._cluster_by_gap(np.array(lightcurve_tab['mjd']),
                                               max_gap=max_gap)
        new_tab = Table(names=lightcurve_tab.colnames, dtype=lightcurve_tab.dtype)

        for cluster_id in np.unique(clusters):
            window_tab = lightcurve_tab[clusters == cluster_id]

            row_values = {}
            for col in lightcurve_tab.colnames:
                if col in band_cols:
                    continue
                col_data = np.array(window_tab[col], copy=True)
                if np.issubdtype(col_data.dtype, np.number):
                    row_values[col] = np.nanmean(col_data)
                else:
                    row_values[col] = col_data[0] if len(col_data) > 0 else None

            for band in bands:
                mag = np.array(window_tab[f'{band}_mag'], dtype=float)
                magerr = np.array(window_tab[f'{band}_magerr'], dtype=float)
                detected = ~np.isnan(magerr)
                use = detected if np.any(detected) else ~np.isnan(mag)

                if f'{band}_snr' in lightcurve_tab.colnames:
                    snr = np.array(window_tab[f'{band}_snr'], dtype=float)
                    row_values[f'{band}_snr'] = (np.nanmean(snr[use])
                                                 if np.any(use) else np.nan)

                if not np.any(detected):
                    # Nothing to inverse-variance weight -- upper limits carry no
                    # per-point uncertainty -- so average the raw limit magnitudes.
                    row_values[f'{band}_mag'] = (np.nanmean(mag[use])
                                                 if np.any(use) else np.nan)
                    row_values[f'{band}_magerr'] = np.nan
                    continue

                m, e = mag[use], magerr[use]
                flux = 10 ** (-m / 2.5)
                flux_err = e * flux * np.log(10) / 2.5
                w = 1.0 / flux_err ** 2
                f_mean = np.sum(w * flux) / np.sum(w)
                f_err_formal = 1.0 / np.sqrt(np.sum(w))

                # AllWISE also tracks the empirical scatter across exposures alongside
                # the formal error, since the latter alone can understate the true
                # uncertainty for a genuinely variable source ("comparison of population
                # variance with composite WPRO flux variance may be used to assess time
                # variability" -- sec4_4c). We report whichever is larger, so this is
                # deliberately not the smallest available number.
                if len(flux) >= 2:
                    f_err_empirical = np.std(flux, ddof=1) / np.sqrt(len(flux))
                else:
                    f_err_empirical = 0.0
                f_err = max(f_err_formal, f_err_empirical)

                row_values[f'{band}_mag'] = -2.5 * np.log10(f_mean)
                row_values[f'{band}_magerr'] = (2.5 / np.log(10)) * f_err / f_mean

            new_tab.add_row([row_values[col] for col in lightcurve_tab.colnames])

        return new_tab
    @staticmethod
    def _bin_stack(
        mag: np.ndarray,
        magerr: np.ndarray,
        mjd: np.ndarray,
        window: float = 365,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inverse-variance weighted stack into non-overlapping bins of `window` days.

        Returns (mjd_centers, mag_stack, magerr_stack).
        """
        flux = 10 ** (-mag / 2.5)
        flux_err = magerr * flux * np.log(10) / 2.5
        bin_edges = np.arange(mjd.min(), mjd.max() + window, window)
        centers, mags, errs = [], [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (mjd >= lo) & (mjd < hi)
            if mask.sum() == 0:
                continue
            w = 1.0 / flux_err[mask] ** 2
            f_mean = (w * flux[mask]).sum() / w.sum()
            f_err = 1.0 / np.sqrt(w.sum())
            centers.append((lo + hi) / 2)
            mags.append(-2.5 * np.log10(f_mean))
            errs.append(2.5 / np.log(10) * f_err / f_mean)
        return np.array(centers), np.array(mags), np.array(errs)

    def get_catalog_lc(self, catalog: str) -> Optional[Table]:
        """The light curve for one catalog, from the local store or the service.

        The store is asked first and may decline (`None`) -- for a catalog it
        does not hold, or a position the release does not contain -- in which
        case this falls through to the live query rather than reporting an
        empty light curve.
        """
        if self.store is not None:
            shaped = self.store.catalog_lc(
                catalog, source_uuid=self.source_uuid,
                ra=self.ra, dec=self.dec,
                radius_arcsec=self.query_rad_arcsec)
            if shaped is not None:
                if len(shaped) == 0:
                    return Table()
                return self._finish_catalog_lc(catalog, shaped)
            print(f'Local store has no {catalog}; falling back to the service.')
        if catalog == 'ztforce':
            from ztforce import run_forced_photometry_batch, build_config
            OUT_DIR = followup.ZTFORCE_CACHE_DIR
            config = build_config()
            all_lcs = run_forced_photometry_batch(
                targets=[self.skycoord],
                bands=('g', 'r', 'i'),
                data_dir=OUT_DIR,
                config=config,
            )
            lcs = all_lcs[0]
            rows = []
            for band, lc in lcs.items():
                det = lc.df[lc.df['detection']]
                if det.empty:
                    continue
                t_centers, mag_stack, magerr_stack = self._bin_stack(
                    mag=det['mag'].to_numpy(),
                    magerr=det['mag_err'].to_numpy(),
                    mjd=det['obsjd'].to_numpy() - 2_400_000.5,
                )
                for mjd_c, m, e in zip(t_centers, mag_stack, magerr_stack):
                    rows.append({'mjd': mjd_c, f'ztforce_{band}_mag': m, f'ztforce_{band}_magerr': e})
            if not rows:
                return Table()
            tab = Table.from_pandas(pd.DataFrame(rows))
            for col in tab.colnames:
                if tab[col].dtype.kind in 'f':
                    tab[col] = np.ma.masked_invalid(tab[col])
            return tab

        if catalog == 'zubercal':
            import lsdb
            import hats.io.file_io.file_io as _hats_fio
            _orig_fits = _hats_fio.read_fits_image
            # Skip the 50 MB point_map.fits download that always times out on cold open.
            # The MOC is only used for query-planning; cone searches work without it.
            _hats_fio.read_fits_image = lambda p, *a, **kw: (
                None if 'point_map' in str(p) else _orig_fits(p, *a, **kw)
            )
            try:
                cat = lsdb.open_catalog(
                    'https://data.lsdb.io/hats/ztf_dr16/zubercal/',
                    columns=['mjd', 'band', 'mag', 'magerr', 'objectid'],
                )
                df = cat.cone_search(self.ra, self.dec, self.query_rad_arcsec).compute()
            finally:
                _hats_fio.read_fits_image = _orig_fits
            if df.empty:
                return Table()
            df = df.copy()
            df['magerr'] = df['magerr'] / 10_000.0
            rows = []
            for b in df['band'].unique():
                bdf = df[df['band'] == b].sort_values('mjd')
                t_centers, mag_stack, magerr_stack = self._bin_stack(
                    mag=bdf['mag'].to_numpy(dtype=float),
                    magerr=bdf['magerr'].to_numpy(dtype=float),
                    mjd=bdf['mjd'].to_numpy(dtype=float),
                )
                for mjd_c, m, e in zip(t_centers, mag_stack, magerr_stack):
                    rows.append({'mjd': mjd_c, f'zubercal_{b}_mag': m, f'zubercal_{b}_magerr': e})
            if not rows:
                return Table()
            tab = Table.from_pandas(pd.DataFrame(rows))
            for col in tab.colnames:
                if tab[col].dtype.kind in 'f':
                    tab[col] = np.ma.masked_invalid(tab[col])
            return tab

        # Construct the lightcurve
        if catalog == 'ztf':
            if self.ztf_local_dir is not None:
                print(f"Querying LOCAL {catalog} catalog for light curve...")
                local_lc = ZTF_LC(
                    self.ztf_local_dir,
                    ra=self.ra,
                    dec=self.dec,
                    query_rad_arcsec=self.query_rad_arcsec,
                )
                lightcurve_tab = local_lc.get_lc()
                lightcurve_tab.rename_column('hmjd', 'mjd')
                return lightcurve_tab

            # Desired column names
            desired_colnames = ['oid', 'ra', 'dec', 'mjd', 'filtercode', 'mag', 'magerr']

            # Rename columns
            column_rename_map = [('oid', 'ztf_id')]

        elif catalog == 'wise':
            # Desired column names
            mag_colnames = [f'w{band_num}mpro_ep' for band_num in range(1, 5)]
            magerr_colnames = [f'w{band_num}sigmpro_ep' for band_num in range(1, 5)]
            f_colnames = [f'w{band_num}flux_ep' for band_num in range(1, 5)]
            ferr_colnames = [f'w{band_num}sigflux_ep' for band_num in range(1, 5)]
            desired_colnames = ['source_id_mf', 'ra', 'dec', 'mjd'] + [
                item for sublist in list(zip(
                    mag_colnames, magerr_colnames, f_colnames, ferr_colnames,
                )) for item in sublist
            ]

            # Rename columns
            column_rename_map = [('source_id_mf', 'wise_id')] + [(f'w{band_num}mpro_ep', f'w{band_num}_mag') for band_num in range(1, 5)] + \
                [(f'w{band_num}sigmpro_ep', f'w{band_num}_magerr') for band_num in range(1, 5)]

        elif catalog == 'neowise':
            # Desired column names
            mag_colnames = [f'w{band_num}mpro' for band_num in range(1, 3)]
            magerr_colnames = [f'w{band_num}sigmpro' for band_num in range(1, 3)]
            snr_colnames = [f'w{band_num}snr' for band_num in range(1, 3)]
            desired_colnames = ['source_id', 'ra', 'dec', 'mjd'] + [
                item for sublist in list(zip(
                    mag_colnames, magerr_colnames, snr_colnames
                )) for item in sublist
            ]

            # Rename columns
            column_rename_map = [('source_id', 'neowise_id')] + [(f'w{band_num}mpro', f'w{band_num}_mag') for band_num in range(1, 3)] + \
                [(f'w{band_num}sigmpro', f'w{band_num}_magerr') for band_num in range(1, 3)] + \
                [(f'w{band_num}snr', f'w{band_num}_snr') for band_num in range(1, 3)]

        elif catalog == 'ptf':
            # Desired column names
            desired_colnames = ['oid', 'ra', 'dec', 'obsmjd', 'fid', 'mag_autocorr', 'magerr_auto'] 

            # Rename columns
            column_rename_map = [('oid', 'ptf_id'), ('obsmjd', 'mjd')]

        elif catalog == 'sdss':
            # Desired column names
            sdss_bands = ['u', 'g', 'r', 'i', 'z']
            desired_colnames = ['objID', 'ra', 'dec', 'mjd'] + [item for sublist in zip(
                [f'psfMag_{band}' for band in sdss_bands],
                [f'psfMagerr_{band}' for band in sdss_bands],
                [f'modelMag_{band}' for band in sdss_bands],
                [f'modelMagerr_{band}' for band in sdss_bands],
            ) for item in sublist]

            # Rename columns
            column_rename_map = [('objID', 'sdss_id')] + [(f'psfMag_{b}', f'{b}Psf_mag') for b in sdss_bands] + \
                [(f'psfMagerr_{b}', f'{b}Psf_magerr') for b in sdss_bands] + \
                [(f'modelMag_{b}', f'{b}Model_mag') for b in sdss_bands] + \
                [(f'modelMagerr_{b}', f'{b}Model_magerr') for b in sdss_bands]

        elif catalog == 'panstarrs':
            # Desired columns
            desired_colnames = ['objID', 'ra', 'dec', 'obsTime', 'infoFlag2'] + [
                    item for sublist in zip(
                        [f'pstarr_{band}_mag' for band in 'grizy'],
                        [f'pstarr_{band}_magerr' for band in 'grizy'],
                    ) for item in sublist
                ]

            # Rename columns
            column_rename_map = [('objID', 'panstarrs_id'), ('obsTime', 'mjd'), ('infoFlag2', 'pstarr_infoFlag2')]

        elif catalog == 'gaia':
            # Desired columns
            gaia_bands = ('g', 'rp', 'bp')
            desired_colnames = ['source_id', 'ra', 'dec', 'ref_epoch'] + [item for sublist in zip(
                [f'phot_{band}_mean_flux' for band in gaia_bands],
                [f'phot_{band}_mean_flux_error' for band in gaia_bands],
            ) for item in sublist]

            # Rename columns
            column_rename_map = [('source_id', 'gaia_id')]
        
        elif catalog == 'custom':
            # Desired columns
            desired_colnames = ['custom_id', 'mjd'] + [item for sublist in zip(
                [f'custom_{band}_mag' for band in 'griz'],
                [f'custom_{band}_magerr' for band in 'griz'],
            ) for item in sublist]

            # Rename columns
            column_rename_map = []

        # Query the catalog
        print(f"Querying {catalog} catalog for light curve...")
        max_attempts = 3
        for i_attempt in range(max_attempts):
            try:
                if catalog == 'ztf':
                    # See Section 10.3 for BAD_CATFLAG_MASK info
                    # https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_explanatory_supplement.pdf
                    lightcurve_tab = Table.from_pandas(
                        lightcurve.LCQuery.from_position(
                            self.ra,
                            self.dec,
                            self.query_rad_arcsec,
                            BAD_CATFLAGS_MASK=BAD_CATFLAG_MASK,
                        ).data
                    )
                    lightcurve_tab = lightcurve_tab[desired_colnames]
                elif catalog == 'panstarrs':

                    # 0 means "no ID": cross_match.py fills unmatched PanSTARRS
                    # IDs with 0 under a mask, and HDF5 drops the mask on write.
                    if (
                        self.pstarr_objid is not None
                        and not np.ma.is_masked(self.pstarr_objid)
                        and not np.isnan(self.pstarr_objid)
                        and int(self.pstarr_objid) != 0
                    ):
                        lightcurve_tab = get_pstarr_lc_from_id(self.pstarr_objid)
                    else:
                        print('Pan-STARRS objid not found. Falling back on coordinate query...')
                        
                        # Sometimes we have a more precise panstarrs coordinate
                        if self.pstarr_coord is None:
                            ra, dec = self.ra, self.dec
                        else:
                            ra, dec = self.pstarr_coord
                        lightcurve_tab = get_pstarr_lc_from_coord(ra, dec, rad_arcsec=self.query_rad_arcsec)
                elif catalog == 'sdss':
                    lightcurve_tab: Table = throttled_sdss_query(
                        SDSS.query_crossid,
                        SkyCoord(self.ra, self.dec, unit='deg'),
                        photoobj_fields=desired_colnames,
                    )
                    # query_crossid can return an HTML error page (e.g. 503) parsed as a
                    # table instead of raising; detect this and treat as no result.
                    if lightcurve_tab is not None and 'objID' not in lightcurve_tab.colnames:
                        print(f'SDSS query_crossid returned unexpected columns {lightcurve_tab.colnames} — treating as no result.')
                        lightcurve_tab = None
                elif catalog == 'custom':
                    # Our own photometry: the nearest follow-up target that has some.
                    targets = [t for t in followup.iter_targets() if t.photometry_file is not None]
                    lightcurve_tab = None
                    if targets:
                        seps = self.skycoord.separation(SkyCoord(
                            [t.ra for t in targets], [t.dec for t in targets], unit='deg'))
                        if np.min(seps.arcsec) <= self.query_rad_arcsec:
                            target = targets[int(np.argmin(seps.arcsec))]
                            lightcurve_tab = pd.read_csv(
                                target.photometry_file,
                                sep=r'\s+',
                                comment='#',
                                names=['mjd', 'custom_mag', 'custom_magerr', 'filter', 'ul', 'telescope', 'instrument']
                            )
                            lightcurve_tab['custom_id'] = target.slug
                            lightcurve_tab = Table.from_pandas(lightcurve_tab)
                else:
                    lightcurve_tab: Table = Irsa.query_region(
                        coordinates=self.skycoord,
                        catalog=self.catalog_astroquery_map[catalog],
                        spatial='Cone',
                        radius=self.query_rad_arcsec * u.arcsec,
                        columns=','.join(desired_colnames),
                    )

                # Escape the loop if no error was raised
                break
            except Exception as e:
                if i_attempt >= max_attempts - 1:
                    raise RuntimeError(
                        f"Failed to fetch light curve from catalog '{catalog}' "
                        f"after {max_attempts} attempt(s)"
                    ) from e
                else:
                    print(f'Attempt {i_attempt + 1} / {max_attempts} to query for light curves failed. Trying again...')
                    traceback.print_exc()

        # Add the wise SNRs and drop the flux columns
        if catalog == 'wise':
            for band_num in range(1, 5):
                f_colname, ferr_colname = f'w{band_num}flux_ep', f'w{band_num}sigflux_ep'
                lightcurve_tab[f'w{band_num}_snr'] = lightcurve_tab[f_colname] / lightcurve_tab[ferr_colname]
                lightcurve_tab.remove_columns([f_colname, ferr_colname])

        # Adjustment for PTF catalog
        if catalog == 'ptf':
            # fid Filter identifier (1 = g; 2 = R).
            # g band
            g_mask = lightcurve_tab['fid'] == 1
            lightcurve_tab['G_mag'] = np.ones(len(lightcurve_tab)) * np.nan
            lightcurve_tab['G_magerr'] = np.ones(len(lightcurve_tab)) * np.nan
            lightcurve_tab['G_mag'][g_mask] = lightcurve_tab['mag_autocorr'][g_mask]
            lightcurve_tab['G_magerr'][g_mask] = lightcurve_tab['magerr_auto'][g_mask]

            # R band
            R_mask = lightcurve_tab['fid'] == 2
            lightcurve_tab['R_mag'] = np.ones(len(lightcurve_tab)) * np.nan
            lightcurve_tab['R_magerr'] = np.ones(len(lightcurve_tab)) * np.nan
            lightcurve_tab['R_mag'][R_mask] = lightcurve_tab['mag_autocorr'][R_mask]
            lightcurve_tab['R_magerr'][R_mask] = lightcurve_tab['magerr_auto'][R_mask]
        elif catalog == 'panstarrs':  # filterID (1=g, 2=r, 3=i, 4=z, 5=y)
            # Iterate through bands and add band magnitude columns
            for i, band in enumerate('grizy'):

                # Get a mask for the filter ID
                filter_id = i + 1
                filter_mask = lightcurve_tab['filterID'].astype(int) == filter_id

                # Fill in the table
                lightcurve_tab[f'pstarr_{band}_mag'] = np.ones(len(lightcurve_tab)) * np.nan
                lightcurve_tab[f'pstarr_{band}_magerr'] = np.ones(len(lightcurve_tab)) * np.nan
                lightcurve_tab[f'pstarr_{band}_mag'][filter_mask] = lightcurve_tab['mag'][filter_mask]
                lightcurve_tab[f'pstarr_{band}_magerr'][filter_mask] = lightcurve_tab['magerr'][filter_mask]

            # Drop unnecessary columns
            lightcurve_tab = lightcurve_tab[desired_colnames]
        elif catalog == 'ztf':
            for band in 'gri':
                band_mask = lightcurve_tab[f'filtercode'] == f'z{band}'
                lightcurve_tab[f'{band}_mag'] = np.ones(len(lightcurve_tab)) * np.nan
                lightcurve_tab[f'{band}_magerr'] = np.ones(len(lightcurve_tab)) * np.nan
                lightcurve_tab[f'{band}_mag'][band_mask] = lightcurve_tab['mag'][band_mask]
                lightcurve_tab[f'{band}_magerr'][band_mask] = lightcurve_tab['magerr'][band_mask]
        elif catalog == 'gaia':
            # Convert time column to mjd
            lightcurve_tab['mjd'] = Time(lightcurve_tab['ref_epoch'], format='jyear', scale='tcb').mjd
            lightcurve_tab.remove_column('ref_epoch')

            # Add magnitude columns
            flux_cols = [f'phot_{band}_mean_flux' for band in gaia_bands]
            mag_cols = [f'gaia_{band}_mag' for band in gaia_bands]
            for band, flux_colname, mag_colname in zip(gaia_bands, flux_cols, mag_cols):
                mag, magerr = img_flux_to_ab_mag(
                    lightcurve_tab[flux_colname],
                    fluxerr=lightcurve_tab[f'{flux_colname}_error'],
                    zero_point=GAIA_ZERO_PTS[band],
                )  # zp from https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photProc/cu5pho_ssec_photCal.html#SSS3.P2
                lightcurve_tab[mag_colname] = mag
                lightcurve_tab[f'{mag_colname}err'] = magerr

            # Drop flux and flux error columns
            lightcurve_tab.remove_columns(flux_cols + [f'{col}_error' for col in flux_cols])
        elif catalog == 'custom' and lightcurve_tab is not None:
            # Add magnitude columns by band
            for band in 'griz':
                band_mask = lightcurve_tab['filter'] == band
                lightcurve_tab[f'custom_{band}_mag'] = np.ones(len(lightcurve_tab)) * np.nan
                lightcurve_tab[f'custom_{band}_magerr'] = np.ones(len(lightcurve_tab)) * np.nan
                lightcurve_tab[f'custom_{band}_mag'][band_mask] = lightcurve_tab['custom_mag'][band_mask]
                lightcurve_tab[f'custom_{band}_magerr'][band_mask] = lightcurve_tab['custom_magerr'][band_mask]

            # Drop unnecessary columns
            lightcurve_tab = lightcurve_tab[desired_colnames]

        if lightcurve_tab is None:
            return Table()

        # Rename columns
        for (cname, new_cname) in column_rename_map:
            lightcurve_tab.rename_column(cname, new_cname)

        return self._finish_catalog_lc(catalog, lightcurve_tab)

    def _finish_catalog_lc(self, catalog: str, lightcurve_tab: Table) -> Table:
        """Shared tail of `get_catalog_lc`: mask, collapse, bin.

        Split out so the local store re-enters here rather than
        reimplementing it. Everything above this point is *fetching and
        naming* -- which differs by definition between a live query and a
        local read -- and everything from here down is the science: the
        Pan-STARRS same-night collapse and the WISE/NEOWISE visit binning,
        including the inverse-variance flux combination. Those must be one
        implementation, not two.
        """
        # Mask NaN values
        for col in lightcurve_tab.colnames:
            if lightcurve_tab[col].dtype.kind in 'f':  # check if the column is of float type
                lightcurve_tab[col] = np.ma.masked_invalid(lightcurve_tab[col])

        # Collapse two measurements from the same night. Pan-STARRS single-epoch measurements visit the same spot
        # twice every night so we can just count that as one measurement
        if catalog == 'panstarrs':

            # The mag and magerr columns
            pstarr_mag_cols = [f'pstarr_{band}_mag' for band in 'grizy']
            pstarr_magerr_cols = [f'pstarr_{band}_magerr' for band in 'grizy']

            # Iterate through the tables of different panstarrs magnitudes
            collapsed_tab = Table(names=lightcurve_tab.columns, dtype=lightcurve_tab.dtype)
            for mag_col, magerr_col in zip(pstarr_mag_cols, pstarr_magerr_cols):
                mag_tab = lightcurve_tab[~lightcurve_tab.mask[mag_col]]

                # Collapse
                while len(mag_tab) > 0:
                    row = mag_tab[0]
                    dt = np.abs(row['mjd'] - mag_tab['mjd'])
                    same_night_mask = (0 < dt) & (dt <= 1)

                    # If there's another observation that night, combine into one row and drop the second one
                    if np.sum(same_night_mask) > 0:

                        # Take the mean of the mag columns and propagate their error
                        row['mjd'] = np.nanmean(np.hstack((row['mjd'], mag_tab[same_night_mask]['mjd'])))
                        row[mag_col] = np.nanmean(np.hstack((row[mag_col], mag_tab[same_night_mask][mag_col])))
                        row[magerr_col] = 1 / (np.sum(same_night_mask) + 1) * np.sqrt(
                            np.nansum(
                                np.hstack((row[magerr_col]**2, mag_tab[same_night_mask][magerr_col]**2))
                            )
                        )  # 1/n * sqrt(sum(sigma_i^2))

                        # Drop the other observation
                        mag_tab = mag_tab[~same_night_mask]

                    # Stack and drop the row
                    collapsed_tab = vstack((collapsed_tab, row))
                    mag_tab.remove_row(0)
            lightcurve_tab = collapsed_tab

        # Ensure mjd column is float dtype
        lightcurve_tab['mjd'] = lightcurve_tab['mjd'].astype(float)

        # WISE and NEOWISE are binned into same-visit exposure clusters (gap <= 5 days
        # between consecutive exposures; see _cluster_by_gap). A visit is documented at
        # ~1.5 days (~12 exposures over ~36 hours), but real per-source clusters can run
        # a bit longer (observed up to ~3.6 days between exposures in the same visit);
        # 5 days gives margin while staying far below the ~180-day revisit cadence.
        #
        # Both go through the same combination -- see _bin_visit_clusters. WISE used to
        # take a plain np.nanmean of every numeric column, which averaged the magerr
        # column and so reported single-exposure precision as if it were the precision
        # of the binned visit; the two surveys were also being combined by different
        # rules within the same plot.
        if catalog == 'wise' and len(lightcurve_tab) > 0:
            lightcurve_tab = self._bin_visit_clusters(
                lightcurve_tab, [f'w{n}' for n in range(1, 5)], max_gap=5.0)

        if catalog == 'neowise' and len(lightcurve_tab) > 0:
            lightcurve_tab = self._bin_visit_clusters(
                lightcurve_tab, [f'w{n}' for n in range(1, 3)], max_gap=5.0)

        return lightcurve_tab


    def get_lc(self) -> Table:
        """Get the light curve."""
        # Query the lightcurves for each catalog in parallel
        lcs: List[Table] = []
        def get_lc_from_cat(catalog_name):
            # Grab the data
            cat_lc = self.get_catalog_lc(catalog_name)

            # Data only for the closest object
            if len(cat_lc) > 0:

                # ZTF uses different oids for different bands, so we must associate between bands
                # we will use an angular separation of <1.5arcseconds as our between-band requirement
                if catalog_name == 'ztf':
                    src_coord = SkyCoord(np.nanmean(cat_lc['ra']), np.nanmean(cat_lc['dec']), unit='deg')
                    all_coords = SkyCoord(cat_lc['ra'], cat_lc['dec'], unit='deg')
                    mask = src_coord.separation(all_coords).arcsec < 1.5
                    cat_lc = cat_lc[mask]
                elif catalog_name not in ('ztforce', 'zubercal', 'neowise'):
                    # ztforce/zubercal already return only the target source's data. NEOWISE
                    # (neowiser_p1bs_psd) is a single-exposure detection catalog with a unique
                    # ID per epoch rather than a persistent per-source ID, so filtering to rows
                    # matching the first epoch's ID silently keeps only ~1 of many real epochs.
                    # The upstream cone search (query_rad_arcsec, well under WISE's ~6" PSF) is
                    # already tight enough that every returned row is the same physical source.
                    mask = cat_lc[f'{catalog_name}_id'] == cat_lc[f'{catalog_name}_id'][0]
                    cat_lc = cat_lc[mask]
                return cat_lc
            return None

        if self.query_in_parallel:
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(get_lc_from_cat, self.catalogs))
        else:
            results = [get_lc_from_cat(cat) for cat in self.catalogs]

        # Filter out None results
        lcs = [result for result in results if result is not None]

        # Join the tables
        if len(lcs) == 1:
            tab = Table(lcs[0], masked=True)
        elif len(lcs) > 1:
            tab = vstack(lcs)
        else:
            tab = Table(names=ALL_LC_COLNAMES, masked=True)

        # Drop unecessary colnames
        for colname in tab.colnames:
            if not colname.endswith('_id') and not colname.endswith('_mag') and not colname.endswith('_magerr') and \
                not colname.endswith('_snr') not in ['mjd', 'ra', 'dec']:
                tab.remove_column(colname)

        # Reorder columns for convenience
        ordered_cols = tab.colnames
        for colname in ordered_cols:
            if colname.endswith('_id'):
                ordered_cols.remove(colname)
                ordered_cols.insert(0, colname)
        tab = tab[ordered_cols]

        return tab
