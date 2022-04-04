"""
Imaging sub-ontology (subset of the EDAM ontology) used for Napari plugins.
Curated synonyms for ontology terms.
"""

import csv
import json
from typing import Any, Dict, List, Optional, Set, Union

from .ontology import Ontology, OntologyTerm


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class CuratedTerm(OntologyTerm):
    def __init__(self, term: OntologyTerm):
        super(CuratedTerm, self).__init__(term.termid, term.name, definition=term.definition)

        self.synonyms = term.synonyms
        self.synonyms_exact = term.synonyms_exact
        self.synonyms_broad = term.synonyms_broad
        self.synonyms_narrow = term.synonyms_narrow
        self.synonyms_related = term.synonyms_related

        self._parent_ids = term.parent_ids
        self._child_ids = term.child_ids

        self.ignore = term.ignore

        # New fields in this class

        self.curated_synonyms_full = set()
        self.curated_synonyms_partial = set()
        self.curated_acronyms = set()

        return
# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def get_imaging_subontology(edam_ontology_tsv: str, imgont_options: Union[str, Dict[str, Any]],
                            verbose: bool = True) -> Ontology:
    """
    Get the imaging sub-ontology containing only terms to recognize in Napari Plugins

    :param edam_ontology_tsv: Path to EDAM Ontology TSV file

    :param imgont_options: Either Dict, or Path to JSON file indicating EDAM Ontology sub-roots and ignored terms.
        Dict with keys:
            - "subroots": Dict[str, str], where the Keys are EDAM ClassID's
            - "ignore_terms": (optional) Dict[str, str], where the Keys are EDAM ClassID's

    :param verbose:
    """

    edam_ont = Ontology.from_edam_tsv(edam_ontology_tsv, verbose=verbose)

    if verbose:
        print()
        print("Reading imaging sub-ontology options from:", imgont_options)

    if isinstance(imgont_options, str):
        with open(imgont_options) as f:
            imgont_options = json.load(f)

    ignore_terms = None
    if (v := imgont_options.get("ignore_terms")) is not None:
        ignore_terms = v.keys()

    img_ont = edam_ont.get_subontology(selected_subroots=imgont_options["subroots"].keys(),
                                       ingore_terms=ignore_terms)

    return img_ont


def read_curated_synonyms(curated_syns_csv: str, verbose: bool = True) -> Dict[str, Dict[str, Union[bool, Set[str]]]]:
    """
    Read the curated-synonyms CSV file and return as a Dict.
        First row is Header with column names.
        Required Columns are:
            ClassID, ..., Excluded?, curated_synonyms_full, curated_acronyms, curated_synonyms_partial
        Blank line terminates entries. Following lines are meta data.

    Returns: Dict
        ClassID => {ignore: bool,
                    curated_synonyms_full: Set[str],
                    curated_acronyms: Set[str],
                    curated_synonyms_partial: Set[str]
                    }
    """
    colnames = []
    fi_classid: Optional[int] = None
    fi_excluded: Optional[int] = None
    fi_curated_synonyms_full: Optional[int] = None
    fi_curated_acronyms: Optional[int] = None
    fi_curated_synonyms_partial: Optional[int] = None

    # ---
    def set_colnames(flds_):
        nonlocal colnames, fi_classid, fi_excluded
        nonlocal fi_curated_synonyms_full, fi_curated_acronyms, fi_curated_synonyms_partial

        colnames = flds_

        fi_classid = colnames.index("ClassID")
        fi_excluded = colnames.index("Excluded?")
        fi_curated_synonyms_full = colnames.index("curated_synonyms_full")
        fi_curated_acronyms = colnames.index("curated_acronyms")
        fi_curated_synonyms_partial = colnames.index("curated_synonyms_partial")
        return

    def get_col(flds_: List[str], fi: int) -> Optional[str]:
        return flds_[fi].strip() or None

    def get_col_bool(flds_: List[str], fi: int) -> bool:
        val = get_col(flds_, fi)
        if val is None:
            return False
        else:
            return val.casefold() == "x"

    def get_col_valset(flds_: List[str], fi: int) -> Set[str]:
        val = get_col(flds_, fi)
        if val is None:
            return set()
        else:
            return set([v.strip() for v in val.splitlines()])
    # ---

    if verbose:
        print()
        print("Reading curated synonyms from:", curated_syns_csv)

    curated_syns = dict()

    n_terms = 0
    with open(curated_syns_csv) as csvf:
        reader = csv.reader(csvf, delimiter=',', quotechar='"')
        for lc, row in enumerate(reader):
            if lc == 0:
                set_colnames(row)
                continue

            if not row:
                break

            n_terms += 1

            curated_syns[get_col(row, fi_classid)] = \
                dict(ignore=get_col_bool(row, fi_excluded),
                     curated_synonyms_full=get_col_valset(row, fi_curated_synonyms_full),
                     curated_acronyms=get_col_valset(row, fi_curated_acronyms),
                     curated_synonyms_partial=get_col_valset(row, fi_curated_synonyms_partial),
                     )

    if verbose:
        print(f"Nbr terms read = {n_terms:,d}")
        print(f"Nbr unique terms = {len(curated_syns):,d}")
        print("Nbr. terms 'Excluded' = ", len([t for t in curated_syns.values() if t['ignore']]))
        for fld in iter(curated_syns.values()).__next__():
            print(f"Nbr. terms with empty `{fld}` = ",
                  len([t for t in curated_syns.values() if not t['ignore'] and not t[fld]]))

    return curated_syns


def get_curated_imaging_subontology(edam_ontology_tsv: str = '../Data/EDAM-bioimaging_alpha06.tsv',
                                    imgsubont_json: str = '../Data/imaging_subontology.json',
                                    curated_syns_csv: str = '../Data/curated_imaging_synonyms-220316.csv',
                                    verbose: bool = True) -> Ontology:
    """
    Merge the curated synonyms with the imaging sub-ontology, using the additional fields in class `CuratedTerm`.
    :return: Ontology with only the relevant terms, where each term is an instance of CuratedTerm.
    """
    img_ont = get_imaging_subontology(edam_ontology_tsv, imgsubont_json, verbose=verbose)
    curated_syns = read_curated_synonyms(curated_syns_csv, verbose=verbose)

    for classid, termsyns in curated_syns.items():
        curated_term = CuratedTerm(img_ont.get_term(classid))
        img_ont.terms[classid] = curated_term

        curated_term.curated_synonyms_full = termsyns["curated_synonyms_full"]
        curated_term.curated_acronyms = termsyns["curated_acronyms"]
        curated_term.curated_synonyms_partial = termsyns["curated_synonyms_partial"]

        if termsyns["ignore"]:
            curated_term.ignore = True

    # Make remaining terms also CuratedTerm
    n_uncurated = 0
    for tid_ in list(img_ont.terms.keys()):
        if not isinstance(term := img_ont.get_term(tid_), CuratedTerm):
            n_uncurated += 1
            img_ont.terms[tid_] = CuratedTerm(term)

    if verbose:
        print()
        print("Nbr terms in Imaging sub-Ontology =", len(img_ont.terms))
        print("Nbr curated terms    =", len(curated_syns))
        print("Nbr un-curated terms =", n_uncurated)
        print()
        print("Stats:")
        print("------")
        pp_curated_ontology_stats(img_ont)

    return img_ont


def pp_curated_ontology_stats(img_ont: Ontology):
    n_terms = 0
    n_acrs, n_syns, n_partials = 0, 0, 0

    # noinspection PyTypeChecker,PyUnusedLocal
    term: CuratedTerm = None

    for term in img_ont.terms.values():
        if term.ignore:
            continue

        n_terms += 1
        n_acrs += len(term.curated_acronyms)
        n_syns += len(term.curated_synonyms_full | {term.name})
        n_partials += len(term.curated_synonyms_partial)

    print(f"Nbr non-excluded ontology terms = {n_terms:,d}")
    print("Nbr ignored (Excluded) terms    = ",
          len([t for t in img_ont.terms.values() if t.ignore]))
    print(f"Nbr Names + Synonyms = {n_syns:,d}, ... {n_syns/n_terms:.1f} per term")
    print(f"Nbr Acronyms         = {n_acrs:,d}, ... {n_acrs/n_terms:.1f} per term")
    print(f"Nbr Partial Synonyms = {n_partials:,d}, ... {n_partials/n_terms:.1f} per term")
    for attr in ["curated_synonyms_full", "curated_synonyms_partial", "curated_acronyms"]:
        print(f"Nbr. terms with empty `{attr:24s}` =",
              "{:3d}".format(len([t for t in img_ont.terms.values() if not t.ignore and not getattr(t, attr)])))
    print()
    return
