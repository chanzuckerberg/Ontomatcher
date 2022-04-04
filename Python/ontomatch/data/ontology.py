"""
API for reading EDAM ontology.
EDAM Bioimaging ontology is extracted from:
    https://github.com/edamontology/edam-bioimaging
"""

import copy
import re
import sys
from typing import Dict, Generator, Iterable, List, Optional, Set, Union


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class EntityExistsError(Exception):
    """
    Raised by `Ontology.add_entity()` if called with an entity-id that already exists.
    """
    def __init__(self, ent_id):
        super(EntityExistsError, self).__init__(f"Entity id '{ent_id}' already exists in the ontology")
# /


class OntologyTerm:
    """
    A term in the EDAM ontology (https://github.com/edamontology/edam-bioimaging)
    """

    SEQ_SEP = "|"

    ACRONYM_SFX_PATT = re.compile(r"\s+\(([^)]+)\)$")

    def __init__(self,
                 termid: str,
                 name: str,
                 definition: Optional[str] = None,
                 synonyms: Optional[str] = None,
                 synonyms_exact: Optional[str] = None,
                 synonyms_broad: Optional[str] = None,
                 synonyms_narrow: Optional[str] = None,
                 synonyms_related: Optional[str] = None,
                 parent_ids: Optional[str] = None,
                 ):
        """
        Arguments
        ---------
            termid: Unique identifier in Ontology for this Term, aka Entity ID, aka Class ID
            name: Preferred Name for this Term
            synonyms: Synonyms
            definition: Definition

        Corresponding fields in the EDAM ontology TSV file:

            synonyms_exact: http://www.geneontology.org/formats/oboInOwl#hasExactSynonym
            synonyms_broad: http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym
            synonyms_narrow: http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym
            synonyms_related: http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym
        """

        self.termid = termid
        self.name = name
        self.definition = definition

        self.synonyms = self._parse_syns(synonyms)

        self.synonyms_exact = self._parse_syns(synonyms_exact)
        self.synonyms_broad = self._parse_syns(synonyms_broad)
        self.synonyms_narrow = self._parse_syns(synonyms_narrow)
        self.synonyms_related = self._parse_syns(synonyms_related)

        if parent_ids == "http://www.w3.org/2002/07/owl#Thing":
            parent_ids = None
        self._parent_ids: Set[str] = set([] if parent_ids is None else parent_ids.split(self.SEQ_SEP))

        self._child_ids: Set[str] = set()

        # Whether this term should not be recognized
        self.ignore: bool = False

        return

    def _parse_syns(self, fval: Optional[str]) -> Set[str]:
        """
        Parse synonyms field value into Set of synonyms.
        Extract abbreviated-names / potential-acronyms from suffixes.

        Examples of synonyms with Acronym in the suffix:
            'Particle Image Correlation Spectroscopy (PICS)'
            'MALDI imaging mass spectrometry (MALDI-IMS)'
            'Multiview selective plane illumination microscopy (MuViSPIM)'
            'inverted SPIM (iSPIM)'
        """
        return parse_syns(fval, self.SEQ_SEP)

    def add_child(self, child_id: str):
        self._child_ids.add(child_id)
        return

    @property
    def entity_id(self) -> str:
        return self.termid

    @property
    def classid(self) -> str:
        return self.termid

    @property
    def parent_ids(self) -> Set[str]:
        return self._parent_ids

    @property
    def child_ids(self) -> Set[str]:
        return self._child_ids

    def __str__(self):
        child_ids = ",\n                 ".join(sorted(self.child_ids))
        return f"{self.__class__.__name__}('{self.termid}', '{self.name}')\n" + \
               f"  parents  [{len(self.parent_ids)}] = {', '.join(sorted(self.parent_ids))}\n" + \
               f"  children [{len(self.child_ids)}] = {child_ids}"
# /


class Ontology:
    def __init__(self, src: str):
        self.src = src
        self.root_termids = set()
        self.terms: Dict[str, OntologyTerm] = dict()
        return

    def add_term(self, term: OntologyTerm, except_if_exists: bool = True):
        if term.termid in self.terms:
            if except_if_exists:
                raise EntityExistsError(term.termid)
            else:
                return

        self.terms[term.termid] = term
        if not term.parent_ids:
            self.root_termids.add(term.termid)
        return

    def get_term(self, term_id: str) -> Optional[OntologyTerm]:
        return self.terms.get(term_id)

    def get_descendants(self, term: Union[str, OntologyTerm]) -> Generator[OntologyTerm, None, None]:
        """
        Depth-first recursive generator.
        """
        if isinstance(term, str):
            term = self.get_term(term)

        yield term
        for tid in term.child_ids:
            yield from self.get_descendants(tid)

        return

    def get_paths_from_root(self, term: OntologyTerm, _paths=None) -> List[List[str]]:
        """
        `term` may have multiple paths from a root in an ontology where some terms have more than one parent.
        An ontology may also have multiple roots.
        Both are true for the EDAM ontology.

        :param term:

        :param _paths: Local use only

        :return: List of paths: List of Term-ID
            In each path [pid_0, ..., pid_n],
                - pid_0 is a root term in the ontology,
                - pid_{i} is a parent of pid_{i+1}
                - pid_n is a parent of `term`

            If `term` is a root term, then returns the empty List of List: [[]]
        """
        if _paths is None:
            _paths = [[]]

        if term is None or not term.parent_ids:
            return _paths

        paths = []
        for path in _paths:
            for pid in term.parent_ids:
                paths.extend(self.get_paths_from_root(self.get_term(pid), _paths=[[pid] + path]))

        return paths

    def has_ancestor(self, term: Union[str, OntologyTerm], ancestor: Union[str, OntologyTerm]) -> bool:
        """
        True if `ancestor` is a direct ancestor of `term`
        """
        return self.get_min_ancestral_distance(term, ancestor) is not None

    def get_min_ancestral_distance(self, term: Union[str, OntologyTerm], ancestor: Union[str, OntologyTerm])\
            -> Optional[int]:
        """
        Distance = 1    If `ancestor` is parent of `term`
                 = 2    If  `ancestor` is grand-parent of `term`
                 = ...
                 = None if `ancestor` is not an ancestor of `term`
        :return: Min such distance, given that a term may have multiple parents
        """
        if isinstance(term, str):
            term = self.get_term(term)

        if isinstance(ancestor, str):
            ancestor_id = self.get_term(ancestor).termid
        else:
            ancestor_id = ancestor.termid

        max_dist = sys.maxsize

        # ---
        def get_distance(path_to_root, anc_id):
            dist = max_dist
            try:
                dist = len(path_to_root) - path_to_root.index(anc_id)
            except ValueError:
                pass

            return dist
        # ---

        paths_to_root = self.get_paths_from_root(term)

        min_dist = min(get_distance(path_, ancestor_id) for path_ in paths_to_root)

        if min_dist < max_dist:
            return min_dist

        return None

    def set_children(self, strict=True):
        for term in self.terms.values():
            for pid in term.parent_ids:
                parent = self.get_term(pid)
                if parent is None:
                    if strict:
                        raise KeyError(pid)
                    else:
                        continue
                parent.add_child(term.termid)
        return

    def pp_term(self, term: Union[str, OntologyTerm]):
        if isinstance(term, str):
            term = self.get_term(term)

        paths_to_root = self.get_paths_from_root(term)
        print(term)
        if paths_to_root:
            print("  Paths to root:")
            for p in paths_to_root:
                print("   ", " : ".join([self.get_term(pid).name for pid in p]))
        print()
        return

    def get_subontology(self, selected_subroots: Iterable[str], ingore_terms: Iterable[str] = None) -> "Ontology":
        """
        Returns a sub-ontology that contains only relevant terms, with paths to the original root terms.

        :param selected_subroots: Sequence of Term-IDs. Sub-trees rooted at these are included.
                All ancestors are also included, but marked as `ignore`.

        :param ingore_terms: Sequence of Term-IDs. These terms are marked to be ignored from recognition.
        """
        if ingore_terms is None:
            ingore_terms = []

        img_ont = Ontology(self.src)

        # ---
        def copy_term(t: OntologyTerm) -> OntologyTerm:
            newt = copy.deepcopy(t)
            newt._child_ids = set()
            return newt

        def copy_from_tree(root_id: str):
            for t in self.get_descendants(root_id):
                img_ont.add_term(copy_term(t), except_if_exists=False)
            return
        # ---

        # Copy the sub-trees, including their ancestors
        for subroot_id in selected_subroots:
            subroot_term = self.get_term(subroot_id)
            # Add all ancestors, mark them to be ignored
            for path_to_root in self.get_paths_from_root(subroot_term):
                for termid in path_to_root:
                    term = copy_term(self.get_term(termid))
                    term.ignore = True
                    img_ont.add_term(term)

            copy_from_tree(subroot_id)

        # Mark the ignored terms
        for termid in ingore_terms:
            term = img_ont.get_term(termid)
            if term:
                term.ignore = True

        # Remove parent-ids that are not in the ontology, except for self.root_terms
        for term in img_ont.terms.values():
            term._parent_ids = set(pid for pid in term.parent_ids
                                   if pid in img_ont.terms or pid in self.root_termids)

        img_ont.set_children(strict=False)

        return img_ont

    @staticmethod
    def from_edam_tsv(ontology_tsv_file: str, verbose=False) -> "Ontology":
        """
        Ingests the EDAM ontology from TSV files
        """
        colnames = []
        fi_classid = None
        fi_name = None
        fi_defn = None

        fi_parent_ids = None

        fi_synonyms = None
        fi_synonyms_exact = None
        fi_synonyms_broad = None
        fi_synonyms_narrow = None
        fi_synonyms_related = None

        # ---
        def find_col(subname):
            for ci, nm in enumerate(colnames):
                if subname in nm:
                    return ci

            print(f"* Warning: column not found: {subname}")
            return None

        def set_colnames(flds_):
            nonlocal colnames, fi_classid, fi_name, fi_defn, fi_parent_ids
            nonlocal fi_synonyms, fi_synonyms_exact, fi_synonyms_broad, fi_synonyms_narrow, fi_synonyms_related

            colnames = flds_

            fi_classid = find_col("Class ID")
            fi_name = find_col("Preferred Label")
            fi_defn = find_col("Definitions")

            fi_synonyms = find_col("Synonyms")
            fi_synonyms_exact = find_col("hasExactSynonym")
            fi_synonyms_broad = find_col("hasBroadSynonym")
            fi_synonyms_narrow = find_col("hasNarrowSynonym")
            fi_synonyms_related = find_col("hasRelatedSynonym")

            fi_parent_ids = find_col("Parents")

            return

        def get_col(flds_: List[str], fi: Optional[int]) -> Optional[str]:
            if fi is None:
                return None
            else:
                return flds_[fi] or None
        # ---

        if verbose:
            print()
            print("Reading ontology from:", ontology_tsv_file)

        ontology = Ontology(ontology_tsv_file)

        n_terms = 0

        with open(ontology_tsv_file) as f:
            for lc, line in enumerate(f):
                flds = [f.strip() for f in line.rstrip(" \n").split("\t")]
                if lc == 0:
                    set_colnames(flds)
                else:
                    if all(not f for f in flds):
                        continue

                    n_terms += 1

                    term = OntologyTerm(get_col(flds, fi_classid), get_col(flds, fi_name), get_col(flds, fi_defn),
                                        get_col(flds, fi_synonyms), get_col(flds, fi_synonyms_exact),
                                        get_col(flds, fi_synonyms_broad), get_col(flds, fi_synonyms_narrow),
                                        get_col(flds, fi_synonyms_related),
                                        get_col(flds, fi_parent_ids),
                                        )
                    ontology.add_term(term)

        ontology.set_children()

        if verbose:
            print(f"Nbr terms read = {n_terms:,d}")
            print(f"Nbr terms in ontology = {len(ontology.terms):,d}")
            print(f"Nbr root terms = {len(ontology.root_termids):,d}")

        return ontology
# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def parse_syns(fval: Optional[str], seq_sep: str) -> Set[str]:
    """
    Parse synonyms field value into Set of synonyms. Extract acronyms from suffixes.

    Examples of synonyms with Acronym in the suffix:
        'Particle Image Correlation Spectroscopy (PICS)'
        'MALDI imaging mass spectrometry (MALDI-IMS)'
        'Multiview selective plane illumination microscopy (MuViSPIM)'
        'inverted SPIM (iSPIM)'
    """
    if fval is None:
        return set()

    syns = set()
    for syn_ in fval.split(seq_sep):
        syn_ = syn_.strip()
        if not syn_:
            continue

        acr = None

        # If potential acronym included as suffix in synonym, parse it out
        if m := OntologyTerm.ACRONYM_SFX_PATT.search(syn_):
            # Htc test for abbrevated name: Not lower-case, and the prefix is not small (or empty).
            # o/w leave it as it is.
            if not m.group(1).islower() and m.start(0) > 5:
                syn_ = syn_[:m.start(0)]
                acr = m.group(1)

        syns.add(syn_)
        if acr:
            syns.add(acr)

    return syns
