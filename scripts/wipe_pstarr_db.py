"""Wipe the Pan-STARRS CasJobs database: every table under the user in Source_Analysis/casjobs.py.

For when the Pan-STARRS SQL extraction query changes. Check that user before running.

    python scripts/wipe_pstarr_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Source_Analysis.casjobs import MASTCASJOBS  # noqa: E402


def wipe_db():
    for table in MASTCASJOBS.list_tables():
        print(f"Dropping table {table}")
        MASTCASJOBS.drop_table(table)

    assert len(MASTCASJOBS.list_tables()) == 0, "Failed to wipe database!!!"


if __name__ == '__main__':
    wipe_db()
