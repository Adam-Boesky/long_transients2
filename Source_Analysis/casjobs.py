"""MAST CasJobs (PanSTARRS_DR2 context): the shared client and the per-object light-curve queries.

One account, so only one process should talk to CasJobs at a time; the site's PS1 light-curve backfill
(`source_db.lightcurves.panstarrs`) runs its own serial chunks.
"""
import numpy as np
from astropy.table import Table
from mastcasjobs import MastCasJobs

from Extracting.utils import get_credentials, img_flux_to_ab_mag

MAST_CREDENTIAL_FNAME = 'mast_dino_login.txt'
wsid, password = get_credentials(MAST_CREDENTIAL_FNAME)
MASTCASJOBS = MastCasJobs(context="PanSTARRS_DR2", userid=wsid, password=password, request_type='POST')


def _add_pstarr_mag_cols(tab: Table) -> Table:
    mags, magerrs = img_flux_to_ab_mag(tab['psfFlux'], fluxerr=tab['psfFluxErr'], zero_point=8.9)  # 8.9 is used by 
    tab['mag'] = mags
    tab['magerr'] = magerrs

    return tab


def _drop_bad_pstarr_flags(tab: Table) -> Table:
    """Drop bad flags. For flag references see https://outerspace.stsci.edu/display/PANSTARRS/PS1+Detection+Flags."""
    tab = tab[tab['infoFlag'] & 2048 == 0]  # source is thought to be a defect
    tab = tab[tab['infoFlag2'] & 4 == 0]    # weird forced photometry thing

    return tab


def get_pstarr_lc_from_id(objid: int) -> Table:
    if not isinstance(objid, int) and not np.isnan(objid):
        objid = int(objid)

    # Construct the query
    query = f"""
SELECT d.objID, d.ra, d.dec, d.obsTime, d.filterID, d.psfFlux, d.psfFluxErr, d.infoFlag, d.infoFlag2, d.zp FROM Detection d WHERE d.objID = {objid}
"""

    # Get table and add mag cols
    tab = MASTCASJOBS.quick(query, task_name=f"PanSTARRS_lc")
    tab = _drop_bad_pstarr_flags(tab)
    tab = _add_pstarr_mag_cols(tab)

    return tab


def get_pstarr_lc_from_coord(ra: float, dec: float, rad_arcsec: float = 1.0) -> Table:
    # fGetNearestObjEq takes ARCMINUTES, so convert. This used to pass
    # rad_arcsec straight through, making the radius 60x too large.
    # Nearest, not nearby: PS1 duplicates sources at skycell boundaries.
    query = f"""
SELECT d.objID, d.ra, d.dec, d.obsTime, d.filterID, d.psfFlux, d.psfFluxErr, d.infoFlag, d.infoFlag2, d.zp
FROM fGetNearestObjEq({ra}, {dec}, {rad_arcsec / 60.0}) nb
INNER JOIN Detection d on d.objID = nb.objID
"""

    # Get table and add mag cols
    tab = MASTCASJOBS.quick(query, task_name=f"PanSTARRS_lc")
    tab = _drop_bad_pstarr_flags(tab)
    tab = _add_pstarr_mag_cols(tab)

    return tab
