"""
Trie based Entity Matcher, for Exact matching of Normalized text.

Ref: https://github.com/google/pygtrie
"""

from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

import pygtrie

import numpy as np

from utils.misc import dict_set_defaults, get_set_element

from .entity_matcher import EntityNameType, EntityMatcher, BasicRevMappedTokenizer, Token


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------

class NameEntry(NamedTuple):
    entity_id: str
    name_type_preference: int
    name_index: int
# /


class NameMatch(NamedTuple):
    entity_id: str
    name_type_preference: int
    name_index: int

    # Index into token-seq on which matches are computed
    start_idx: int
    # Length of matching key, in nbr tokens. So match is for token-seq[start_idx : start_idx + key_length]
    key_length: int

    # Position of match in source string
    char_start: int
    char_end: int

    @staticmethod
    def from_name_entry(name_entry: NameEntry, start_idx: int, key_length: int, char_start: int, char_end: int):
        return NameMatch(entity_id=name_entry.entity_id,
                         name_type_preference=name_entry.name_type_preference,
                         name_index=name_entry.name_index,
                         start_idx=start_idx,
                         key_length=key_length,
                         char_start=char_start,
                         char_end=char_end
                         )
# /


class TrieMatchHelper:
    """
    Exact matcher on Normalized Names, functioning as a Mapping:
        Key: Sequence[str] = a normalized tokenized name
        Value: Set[NameEntry]
    """

    ROUND_NBR_DECIMALS = 3

    def __init__(self, name: str, min_name_length: int = 1, stop_names: List[str] = None):
        self.name = name

        self.min_name_length = min_name_length
        self.stop_names = set(stop_names) if stop_names else set()

        # self.add_name_type_preference_to_score = add_name_type_preference_to_score

        self.normalized_names: List[List[str]] = []

        # Only used for debugging.
        # original_names[i] = Set of original names, from `self.add_name()`,
        #                       that normalized to `self.normalized_names[i]`
        self.original_names: List[Set[str]] = []

        # List[str] => Set[NameEntry]]
        self.trie = pygtrie.Trie()
        return

    def add_name(self, entity_id: str, original_name: str, normalized_name: List[str], name_type_preference: int):
        """
        :returns: -1 if name not added
        """
        entity_id = entity_id.strip()
        original_name = original_name.strip()

        if not entity_id or not original_name:
            return -1

        if len(normalized_name) == 1:
            if len(normalized_name[0]) < self.min_name_length or normalized_name[0] in self.stop_names:
                return -1

        kvalue = self.trie.get(normalized_name)
        if kvalue is None:
            kvalue = set()
            self.trie[normalized_name] = kvalue

        if kvalue:
            name_index = get_set_element(kvalue).name_index
            self.original_names[name_index].add(original_name)
        else:
            name_index = len(self.normalized_names)
            self.normalized_names.append(normalized_name)
            self.original_names.append({original_name})

        kvalue.add(NameEntry(entity_id, name_type_preference, name_index))
        return name_index

    def get_original_names(self, name_index: int) -> Set[str]:
        return self.original_names[name_index]

    def get_normalized_name(self, name_index: int) -> str:
        return " ".join(self.normalized_names[name_index])

    def get_names(self, name_indices: Iterable[int]) -> Tuple[Set[str], Set[str]]:
        orig_names, norm_names = set(), set()
        for name_index in name_indices:
            orig_names |= self.get_original_names(name_index)
            norm_names.add(self.get_normalized_name(name_index))

        return orig_names, norm_names

    def get_full_matches(self, mention_tkns: List[Token]) -> Set[NameMatch]:
        mention = [t.text for t in mention_tkns]
        name_entries = self.trie.get(mention, [])

        m_char_start = mention_tkns[0].char_start
        m_char_end = mention_tkns[-1].char_end

        return set([NameMatch.from_name_entry(ne, 0, len(mention), m_char_start, m_char_end)
                    for ne in name_entries])

    def get_all_sub_matches(self, mention_tkns: List[Token]) -> Set[NameMatch]:
        """
        Match all sub-sequences of `tokens`.
        """
        tokens = [t.text for t in mention_tkns]
        name_matches = set()
        for s in range(len(tokens)):
            for key, name_ent_set in self.trie.prefixes(tokens[s:]):
                m_char_start = mention_tkns[s].char_start
                m_char_end = mention_tkns[s + len(key) - 1].char_end
                name_matches.update(NameMatch.from_name_entry(ne, s, len(key), m_char_start, m_char_end)
                                    for ne in name_ent_set)

        return name_matches

    def get_prefix_matches(self, mention_tkns: List[Token]) -> Set[NameMatch]:
        """
        Matches to all prefixes of `tokens`.
        """
        tokens = [t.text for t in mention_tkns]
        name_matches = set()
        s = 0
        for key, name_ent_set in self.trie.prefixes(tokens):
            m_char_start = mention_tkns[s].char_start
            m_char_end = mention_tkns[s + len(key) - 1].char_end
            name_matches.update(NameMatch.from_name_entry(ne, s, len(key), m_char_start, m_char_end)
                                for ne in name_ent_set)

        return name_matches

    @staticmethod
    def sorted_matches(name_matches: Set[NameMatch],
                       nmax: int = None, min_score: float = None) \
            -> List[NameMatch]:
        """
        Discard matches below `min_score`,
        sort on (score, match_name_type) descending, and select top `nmax`.
        """

        if not name_matches:
            return []

        name_matches = list(name_matches)

        match_scores = np.asarray([nm.key_length for nm in name_matches])

        if min_score is not None:
            nz_i = np.nonzero(match_scores >= min_score)[0]
            if not nz_i:
                return []
        else:
            nz_i = np.arange(match_scores.shape[0])

        # This will do a compound sort on (score, match_name_type)
        # ASSUMES: name_type_preferences < 100
        # nz = np.around(match_scores[nz_i], self.ROUND_NBR_DECIMALS) * np.power(10, self.ROUND_NBR_DECIMALS + 2)
        # ... Don't need np.around() since match_scores are int here
        nz = match_scores[nz_i] * np.power(10, 2)

        name_type_preferences = np.asarray([name_matches[i].name_type_preference for i in nz_i])
        nz += name_type_preferences

        # Use "mergesort" for stable sort
        if not nmax:
            s_i = np.argsort(-nz, kind="mergesort")
        else:
            s_i = np.argsort(-nz, kind="mergesort")[:nmax]

        sorted_idxs = nz_i[s_i]

        return [name_matches[i] for i in sorted_idxs]

# /


class TrieMatcher(EntityMatcher):

    DEFAULT_PARAMS = {
        "add_name_preference_to_score": False,

        "min_name_length": 2,

        "name_type_preference": {
            EntityNameType.PRIMARY.value: 4,
            EntityNameType.ACRONYM.value: 3,
            EntityNameType.SYNONYM.value: 2,
            EntityNameType.PARTIAL.value: 1
        }
    }

    def __init__(self, params: Dict[str, Any] = None):
        """
        params -- fields:
            - class
            - name
            - descr
            - cache_file: Optional.

            - lexicon_id: str.
                    Used to identify the Lexicon this matcher is built from.
                    Can be a path to the Lexicon file.

            - name_type_preference: Dict[str, int]
                    Assigns integer values to name-types. Higher values represent higher preference.
                    Name types must be from `EntityNameType`. Integer values must be > 0.

            - add_name_preference_to_score: bool, default False.
                    Whether match's name_type_preference value is added to match-score.

            - min_name_length: int, default = 2

            - stop_names: List[str]
                e.g. ["as"]
        """

        if params is None:
            params = dict(lexicon_id="DefaultLexiconID")

        super().__init__(params)
        dict_set_defaults(self.params, self.DEFAULT_PARAMS)

        self.lexicon_id = self.params["lexicon_id"]
        self.cache_file = self.params.get("cache_file")

        # --- These are used only for calculating the score of a match

        self.add_name_preference_to_score = self.params["add_name_preference_to_score"]

        self.name_preference_primary = int(self.params["name_type_preference"][EntityNameType.PRIMARY.value])
        self.name_preference_synonym = int(self.params["name_type_preference"][EntityNameType.SYNONYM.value])
        self.name_preference_acronym = int(self.params["name_type_preference"][EntityNameType.ACRONYM.value])
        self.name_preference_partial = int(self.params["name_type_preference"][EntityNameType.PARTIAL.value])

        assert self.name_preference_primary > 0
        assert self.name_preference_synonym > 0
        assert self.name_preference_acronym > 0
        assert self.name_preference_partial > 0

        # --- Derived fields

        self.tknzr = BasicRevMappedTokenizer()

        # --- Fields populated by build / load

        self.match_helper = TrieMatchHelper("Names and Acronyms",
                                            min_name_length=self.params["min_name_length"],
                                            stop_names=self.params.get("stop_names"))

        return

    def _add_entity_names(self, entity_id: str,
                          primary_name: str, synonyms: Sequence[str], acronyms: Sequence[str],
                          partial_names: Sequence[str] = None):

        if partial_names is None:
            partial_names = []

        syn_idxs = set()

        idx = self.match_helper.add_name(entity_id, primary_name,
                                         [tkn.text for tkn in self.normalize_name(primary_name)],
                                         self.name_preference_primary)
        if idx >= 0:
            syn_idxs.add(idx)

        for name_ in synonyms:
            idx = self.match_helper.add_name(entity_id, name_,
                                             [tkn.text for tkn in self.normalize_name(name_)],
                                             self.name_preference_synonym)
            if idx >= 0:
                syn_idxs.add(idx)

        acr_idxs = set()

        for name_ in acronyms:
            idx = self.match_helper.add_name(entity_id, name_,
                                             [tkn.text for tkn in self.normalize_acronym(name_)],
                                             self.name_preference_acronym)
            if idx >= 0:
                acr_idxs.add(idx)

        partial_idxs = set()

        for name_ in partial_names:
            idx = self.match_helper.add_name(entity_id, name_,
                                             [tkn.text for tkn in self.normalize_name(name_)],
                                             self.name_preference_partial)
            if idx >= 0:
                partial_idxs.add(idx)

        self.entity_info[entity_id].name_indices = {EntityNameType.SYNONYM: syn_idxs,
                                                    EntityNameType.ACRONYM: acr_idxs,
                                                    EntityNameType.PARTIAL: partial_idxs
                                                    }

        return

    def compile(self):
        # Nothing to do
        self.is_compiled = True
        return

    def compile_and_save(self):
        data_cache_file = self.params["cache_file"]
        print('   Data will be cached to:', data_cache_file)

        start_tm = datetime.now()

        data_cache_file = os.path.expanduser(data_cache_file)
        if Path(data_cache_file).exists():
            print('   ... removing existing file', flush=True)
            Path(data_cache_file).unlink()

        # Optimize the structures
        self.compile()

        self.save(data_cache_file)

        end_time = datetime.now()
        print('   Total Build time  =', end_time - start_tm)
        print()

        return

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate the loaded data's options as loaded from `self.load_data_cache()`
        against options specified in `__init__()` during creation of this instance.

        May potentially update `params`
        """
        dict_set_defaults(params, self.DEFAULT_PARAMS)

        for k in self.params:
            if k in {"name", "descr", "cache_file"}:
                continue

            if self.params[k] != params.get(k):
                return False

        return True

    def copy_from(self, another: "TrieMatcher"):
        """
        Copy data populated during `self.build_and_save()` from another instance.
        """
        # noinspection PyUnresolvedReferences
        from grantinfo.lexmatch.triematcher import TrieMatcher

        assert isinstance(another, TrieMatcher), \
            f"type(another) = '{type(another)}', should be '{TrieMatcher.__qualname__}'"

        super().copy_from(another)

        self.match_helper = another.match_helper

        # No need to copy:
        #   ... member fields caching params
        #   self.name_preference_*
        #   ... invariant members
        #   self.tknzr

        return

    def normalize_name(self, name: str):
        """
        Normalize a Concept Name, or a Mention being tested as a Concept Name
        """
        return self.tknzr.tokenize(name, to_lower=True)

    def normalize_acronym(self, acronym: str):
        """
        Normalize a Concept Acronym, or a Mention being tested as a Concept Acronym
        """
        return self.tknzr.tokenize(acronym, to_lower=False)

    def get_normalized_name(self, name_index: int) -> str:
        return self.match_helper.get_normalized_name(name_index)

    def get_original_name(self, name_index: int) -> str:
        raise NotImplementedError

    def get_original_names(self, name_index: int) -> Set[str]:
        """Override when returning multiple names mapping to same name-index."""
        return self.match_helper.get_original_names(name_index)

    def get_unique_synonyms(self, entity_id: str) -> Tuple[List[str], List[str]]:
        syn_idxs = self.entity_info[entity_id].name_indices[EntityNameType.SYNONYM]
        orig_names, norm_names = self.match_helper.get_names(syn_idxs)
        return list(orig_names), list(norm_names)

    def get_unique_partial_names(self, entity_id: str) -> Tuple[List[str], List[str]]:
        syn_idxs = self.entity_info[entity_id].name_indices[EntityNameType.PARTIAL]
        orig_names, norm_names = self.match_helper.get_names(syn_idxs)
        return list(orig_names), list(norm_names)

    def get_unique_acronyms(self, entity_id: str) -> Optional[Tuple[List[str], List[str]]]:
        syn_idxs = self.entity_info[entity_id].name_indices[EntityNameType.ACRONYM]
        if syn_idxs:
            orig_names, norm_names = self.match_helper.get_names(syn_idxs)
            return list(orig_names), list(norm_names)

        return None

    # =====================================================================================================
    #       Methods for entity matching
    # =====================================================================================================

    def get_full_matches(self, text: str) -> List[NameMatch]:
        """
        Entities matching entire text
        """
        tokens = self.normalize_name(text)
        name_matches = self.match_helper.get_full_matches(tokens)
        return self.match_helper.sorted_matches(name_matches)

    def get_all_matching_entities(self, text: str,
                                  nmax: int = None, min_score: float = None) -> List[NameMatch]:
        """
        All Entity matches into text, possibly overlapping.
        """
        tokens = self.normalize_name(text)
        name_matches = self.match_helper.get_all_sub_matches(tokens)

        tokens = self.normalize_acronym(text)
        name_matches.update(self.match_helper.get_all_sub_matches(tokens))

        return self.match_helper.sorted_matches(name_matches, nmax=nmax, min_score=min_score)

    def get_greedy_nonoverlapping_matches(self, text: str,
                                          nmax: int = None, min_score: float = None) -> List[NameMatch]:
        """
        Greedy selection of non-overlapping matches, prefers:
            - early matches
            - longer matching keys (in nbr of tokens)
        """
        name_matches = self.get_all_matching_entities(text, nmax=nmax, min_score=min_score)

        greedy_matches = []
        selected_spans = []

        for nmatch in name_matches:
            s, w = nmatch.start_idx, nmatch.key_length
            if not span_overlaps_selections(s, w, selected_spans):
                selected_spans.append((s, w))
                greedy_matches.append(nmatch)

        return greedy_matches

# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def span_overlaps_selections(s: int, w: int, selected_spans: Iterable[Tuple[int, int]]):
    e = s + w
    for s2, w2 in selected_spans:
        e2 = s2 + w2
        if span_overlaps(s, e, s2, e2):
            return True

    return False


def span_overlaps(s1, e1, s2, e2):
    return s1 < e2 and e1 > s2
