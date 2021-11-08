"""
Reads data from CSV files, downloaded from the Google Sheet:
    https://docs.google.com/spreadsheets/d/1LHhkIq_PUgx2fVblZWwIZb8HvjS_sWN6stgiA-OyZ8U/edit#gid=1394695008
"""

import csv
from typing import Dict, List, NamedTuple, Set, Union


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------

class Plugin(NamedTuple):
    name: str
    summary: str
    description: str
# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def read_plugins(plugins_csv: str) -> List[Plugin]:
    """
    :param plugins_csv: Two file types supported, based on the following two col-names in Row 0:
        samples:    ['name of plugin', 'plugin description - text']
        full dump:  ['', 'name', 'summary', 'description']

    :returns: List[ Plugin, ...]
    """
    c_name, c_descr = 0, 1
    c_summary = None

    entries = []
    with open(plugins_csv, newline="") as csvf:
        reader = csv.reader(csvf, delimiter=',', quotechar='"')
        for lc, row in enumerate(reader):
            if lc == 0:
                if row[1:4] == ['name', 'summary', 'description']:
                    c_name, c_summary, c_descr = 1, 2, 3
                elif row == ['name of plugin', 'plugin description - text']:
                    c_name, c_summary, c_descr = 0, None, 1
                else:
                    raise TypeError("Unrecognized header for file: " + plugins_csv)

                continue

            name = row[c_name].strip()
            if not name:
                continue

            descr = row[c_descr].strip()
            # Remove repeat of plugin-name in descr
            if descr.startswith(name):
                descr = descr[len(name):].strip()

            summary = None
            if c_summary is not None:
                summary = row[c_summary].strip()
                if not summary:
                    summary = None

            entries.append(Plugin(name, summary, descr))

    return entries


def read_ontology(ontology_csv: str) -> Dict[str, Dict[str, Union[str, Set[str]]]]:
    """
    Cols in file are:
        Term Name
        Term partial names  (one per line)
        Synonyms            (one per line)
        Partial Synonyms    (one per line)

    Returns: Dict
        Name => {name: str, synonyms: Set[str], partials: Set[str]}
    """
    entries = dict()
    with open(ontology_csv, newline="") as csvf:
        reader = csv.reader(csvf, delimiter=',', quotechar='"')
        for lc, row in enumerate(reader):
            if lc <= 1:
                continue
            row[0] = row[0].strip()
            if not row[0]:
                continue

            term = dict(name=row[0].strip(),
                        synonyms=set(x.strip() for x in row[2].splitlines()),
                        partials=(set(x.strip() for x in row[1].splitlines())
                                  | set(x.strip() for x in row[3].splitlines()))
                        )

            entries[row[0]] = term

    return entries
