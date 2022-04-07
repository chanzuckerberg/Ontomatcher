"""
Trie based Entity Matcher, for Exact matching of Normalized text.

Ref: https://github.com/google/pygtrie
"""

import dataclasses
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

import pygtrie

import numpy as np

from ontomatch.utils.misc import get_set_element

from .nttmatcher import NameMatch, EntityMatcher
from .tokenizer import BasicRevMappedTokenizer, Token, NormalizationType


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------

class NameEntry(NamedTuple):
    """
    Trie maps each normalized-name to a Set[NameEntry]
    """

    entity_id: str
    name_type: str
    name_tier: int
    name_index: int
# /


class OriginalName(NamedTuple):
    entity_id: str
    name: str
    name_type: str
# /


@dataclasses.dataclass(frozen=True)
class NameMatchImpl(NameMatch):
    # Used in sorting matches
    name_tier: int

    # Index into token-seq on which matches are computed
    start_idx: int
    # Length of matching key, in nbr tokens. So match is for token-seq[start_idx : start_idx + key_length]
    key_length: int

    @staticmethod
    def from_name_entry(name_entry: NameEntry, start_idx: int, key_length: int, char_start: int, char_end: int):
        return NameMatchImpl(entity_id=name_entry.entity_id,
                             name_type=name_entry.name_type,
                             name_tier=name_entry.name_tier,
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

    def __init__(self, name: str, min_name_length: int = 1, stop_names: List[List[str]] = None):
        """

        :param name: Name of this `TrieMatchHelper`
        :param min_name_length: Single-token names with length less than this are rejected
        :param stop_names: List of tokenized-and-normalized names that are considered invalid names (and skipped)
        """
        self.name = name

        self.min_name_length = min_name_length
        # Convert List to Tuple for use in Set
        self.stop_names = set(tuple(nm) for nm in stop_names) if stop_names else set()

        # Each entry is List[normalized-token]
        self.normalized_names: List[List[str]] = []

        # Used to retrieve the original Entity Names that all map to the same normalized-name.
        # original_names[i] = Set of OriginalName's, from `self.add_name()`,
        #                       that normalized to `self.normalized_names[i]`
        self.original_names: List[Set[OriginalName]] = []

        # List[str] => Set[NameEntry]
        self.trie = pygtrie.Trie()
        return

    def add_name(self, entity_id: str, original_name: str, normalized_name: List[str],
                 name_type: str, name_tier: int):
        """
        Add a name for entity `entity_id`.

        :param entity_id: Unique id for this entity
        :param original_name: The original name being added
        :param normalized_name: The normalized name, corresponding to the `original_name`, being added
        :param name_type: One of the Name-Types as specified in `EntityMatcher.name_type_params`
        :param name_tier: Specifies the priority tier for matching this name

        The `normalized_name` is rejected if:
            - It is a single token, and its length < `self.min_name_length`
            - It matches one of the stop-names

        :returns: -1 if rejected,
                Else:
                  Index (>= 0) at which this name is added.
                  This index corresponds to the `NameMatch.name_index` in a match,
                  and can be used in the following methods to retrieve the names:
                    `self.get_original_names()`, `self.get_normalized_name()`, `self.get_names()`
        """
        entity_id = entity_id.strip()
        original_name = original_name.strip()

        if not entity_id or not original_name:
            return -1

        if len(normalized_name) == 1:
            if len(normalized_name[0]) < self.min_name_length:
                return -1
        elif tuple(normalized_name) in self.stop_names:
            return -1

        kvalue = self.trie.get(normalized_name)
        if kvalue is None:
            kvalue = set()
            self.trie[normalized_name] = kvalue

        if kvalue:
            name_index = get_set_element(kvalue).name_index
            self.original_names[name_index].add(OriginalName(entity_id, original_name, name_type))
        else:
            name_index = len(self.normalized_names)
            self.normalized_names.append(normalized_name)
            self.original_names.append({OriginalName(entity_id, original_name, name_type)})

        kvalue.add(NameEntry(entity_id, name_type, name_tier, name_index))
        return name_index

    def nbr_keys(self):
        return len(self.original_names)

    def get_all_entity_ids(self, name_type: Optional[str] = None) -> Set[str]:
        return set([orig_name_.entity_id for orig_names in self.original_names for orig_name_ in orig_names
                    if name_type is None or orig_name_.name_type == name_type])

    def get_original_names(self, name_index: int) -> Set[OriginalName]:
        return self.original_names[name_index]

    def get_normalized_name(self, name_index: int) -> str:
        return " ".join(self.normalized_names[name_index])

    def get_names(self, name_indices: Iterable[int]) -> Tuple[Set[OriginalName], Set[str]]:
        if name_indices is None:
            name_indices = []

        orig_names, norm_names = set(), set()
        for name_index in name_indices:
            orig_names |= self.get_original_names(name_index)
            norm_names.add(self.get_normalized_name(name_index))

        return orig_names, norm_names

    def get_full_matches(self, mention_tkns: List[Token]) -> Set[NameMatchImpl]:
        mention = [t.text for t in mention_tkns]
        name_entries = self.trie.get(mention, set())

        m_char_start = mention_tkns[0].char_start
        m_char_end = mention_tkns[-1].char_end

        return set([NameMatchImpl.from_name_entry(ne, 0, len(mention), m_char_start, m_char_end)
                    for ne in name_entries])

    def get_all_sub_matches(self, mention_tkns: List[Token]) -> Set[NameMatchImpl]:
        """
        Match all sub-sequences of `tokens`.
        """
        tokens = [t.text for t in mention_tkns]
        name_matches = set()
        for s in range(len(tokens)):
            for key, name_ent_set in self.trie.prefixes(tokens[s:]):
                m_char_start = mention_tkns[s].char_start
                m_char_end = mention_tkns[s + len(key) - 1].char_end
                name_matches.update(NameMatchImpl.from_name_entry(ne, s, len(key), m_char_start, m_char_end)
                                    for ne in name_ent_set)

        return name_matches

    def get_prefix_matches(self, mention_tkns: List[Token]) -> Set[NameMatchImpl]:
        """
        Matches to all prefixes of `tokens`.
        """
        tokens = [t.text for t in mention_tkns]
        name_matches = set()
        s = 0
        for key, name_ent_set in self.trie.prefixes(tokens):
            m_char_start = mention_tkns[s].char_start
            m_char_end = mention_tkns[s + len(key) - 1].char_end
            name_matches.update(NameMatchImpl.from_name_entry(ne, s, len(key), m_char_start, m_char_end)
                                for ne in name_ent_set)

        return name_matches

    @staticmethod
    def sorted_matches(name_matches: Set[NameMatchImpl]) -> List[NameMatchImpl]:
        """
        Sort on:
            (name-tier [ascending], key_length [descending], start_token_index [asc.], entity_id [asc.]),
        ... where:
            key_length = Nbr. normalized tokens that matched (i.e. prefer longer matches)
            entity_id used in case multiple entities have the same normalized name
        """

        if not name_matches:
            return []

        # For indexing at the end
        name_matches = list(name_matches)

        # Assemble for Compound Sort ... include enough keys to make sort deterministic

        names_in_order = ["tier", "key_length", "start_idx", "entity_id"]

        match_scores = np.core.records.fromrecords([(nm.name_tier, -nm.key_length, nm.start_idx, nm.entity_id)
                                                    for nm in name_matches],
                                                   names=names_in_order)

        # Use "mergesort" for stable sort
        sorted_idxs = np.argsort(match_scores, kind="mergesort", order=names_in_order)

        return [name_matches[i] for i in sorted_idxs]

# /


class TrieMatcher(EntityMatcher):

    def __init__(self, *args, **kwargs):
        """
        Args: Same as for `EntityMatcher`
        """

        super().__init__(*args, **kwargs)

        # --- Local fields

        self.tknzr = BasicRevMappedTokenizer()

        # --- Fields populated by build / load

        # Separate match-helper for each style of normalization
        self.match_helpers: Dict[NormalizationType, TrieMatchHelper] = \
            {norm_type: TrieMatchHelper(norm_type.value,
                                        min_name_length=self.min_name_length,
                                        stop_names=[self.tknzr.normalized_tokens(name_, normalization_type=norm_type)
                                                    for name_ in self.stop_names])
             for norm_type in NormalizationType
             }

        return

    def matches_loaded_obj(self, loaded_obj: "TrieMatcher") -> bool:
        return super().matches_loaded_obj(loaded_obj)

    def add_entity(self, entity_id: str, primary_name: str, synonyms: Sequence[str], acronyms: Sequence[str],
                   partial_names: Sequence[str] = None):
        """
        Convenience method, using the name-types as defined in `EXAMPLE_PARAMS`
        """

        # Remove primary_name from synonyms if using same normalization
        primary_norm_type = self.name_type_params[self.NAME_TYPE_PRIMARY]["normalization"]

        if "synonym" in self.name_type_params \
                and self.name_type_params["synonym"]["normalization"] == primary_norm_type:

            normlzd_primary = self.tknzr.normalize(primary_name, normalization_type=primary_norm_type)
            synonyms = [syn for syn in synonyms
                        if self.tknzr.normalize(syn, normalization_type=primary_norm_type) != normlzd_primary]

        self.add_entity_with_names(entity_id, primary_name,
                                   dict(synonym=synonyms, acronym=acronyms, partial=partial_names))
        return

    def _add_names_for_entity(self, entity_id: str, name_type: str, names: Sequence[str]):

        tkn_norm_type, name_tier, match_helper = self._get_params_and_match_helper(name_type)

        syn_idxs = set()

        for name_ in names:
            idx = match_helper.add_name(entity_id, name_,
                                        self.tknzr.normalized_tokens(name_, normalization_type=tkn_norm_type),
                                        name_type, name_tier)
            if idx >= 0:
                syn_idxs.add(idx)

        self.entity_info[entity_id].name_indices[name_type].update(syn_idxs)
        return

    def _get_params_and_match_helper(self, name_type: str):
        try:
            name_type_params = self.name_type_params[name_type]
        except KeyError:
            raise KeyError(f"name_type = '{name_type}' not defined for this {self.__class__.__name__} instance")

        # noinspection PyTypeChecker
        tkn_norm_type: NormalizationType = name_type_params["normalization"]
        name_tier = name_type_params["tier"]

        match_helper = self.match_helpers[tkn_norm_type]

        return tkn_norm_type, name_tier, match_helper

    def compile(self):
        # Nothing to do
        self.is_compiled = True
        return

    def compile_and_save(self):
        assert self.cache_file is not None, "`cache_file` path not provided!"

        print('   Data will be cached to:', self.cache_file)

        start_tm = datetime.now()

        data_cache_file = os.path.expanduser(self.cache_file)
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

    def copy_from(self, another: "TrieMatcher"):
        """
        Copy data populated during `self.compile_and_save()` from another instance.
        """
        # noinspection PyUnresolvedReferences
        from ontomatch.text.triematcher import TrieMatcher

        assert isinstance(another, TrieMatcher), \
            f"type(another) = '{type(another)}', should be '{TrieMatcher.__qualname__}'"

        super().copy_from(another)

        self.match_helpers = another.match_helpers

        # No need to copy:
        #   ... invariant members
        #   self.tknzr

        return

    # =====================================================================================================
    #       Stats
    # =====================================================================================================

    def pp_stats(self):
        hdr = f"Stats for {self.__class__.__name__}, lexicon_id = '{self.lexicon_id}':"
        print(hdr)
        print("-" * len(hdr))
        print()

        print(f"Nbr entities = {self.get_nbr_entities():,d}")
        print()

        for name_type in self.name_type_params:
            tkn_norm_type, name_tier, match_helper = self._get_params_and_match_helper(name_type)
            print(f"Name Type = {name_type}, tier = {name_tier}, normalization = {tkn_norm_type.name}")
            print()

            print("   ", f"Nbr entities in this type      = {len(match_helper.get_all_entity_ids(name_type)):5,d}")

            all_names = set(tuple([orig_name_ for orig_name_ in orig_names if orig_name_.name_type == name_type])
                            for orig_names in match_helper.original_names) - {tuple()}

            ambig_names = [orig_names for orig_names in all_names if len(orig_names) > 1]

            print("   ", f"Nbr unique normalized names    = {len(all_names):5,d}")

            # ambig_names = [orig_names
            #                for orig_names in match_helper.original_names
            #                if len([orig_name_ for orig_name_ in orig_names if orig_name_.name_type == name_type]) > 1]

            print("   ", f"Nbr ambiguous normalized names = {len(ambig_names):5,d}")
            print()

        return

    # =====================================================================================================
    #       Methods for retrieving names for an Entity / NameMatch
    # =====================================================================================================

    def get_all_unique_names(self, entity_id: str) -> Dict[str, Tuple[Set[str], Set[str]]]:
        """
        :return: Dict: NameType[str] => Tuple[ Set[Orig Name], Set[Normalized Name] ]
            ... where NameType's are as defined in `self.name_type_params`.
        """
        all_syns = {name_type: self.get_unique_names(entity_id, name_type)
                    for name_type in self.entity_info[entity_id].name_indices}

        return all_syns

    def get_unique_names(self, entity_id: str, name_type: str) -> Tuple[Set[str], Set[str]]:
        syn_idxs = self.entity_info[entity_id].name_indices.get(name_type)
        _, _, match_helper = self._get_params_and_match_helper(name_type)
        orig_names, norm_names = match_helper.get_names(syn_idxs)
        return set([on.name for on in orig_names if on.entity_id == entity_id]), norm_names

    def get_matching_names(self, name_match: NameMatch) -> Tuple[Set[str], str]:
        """
        Get the original (un-normalized) names and the single normalized name associated with a `NameMatch`.
        :return: Set[ Original-Names (un-normalized) ], Normalized-Name
        """
        _, _, match_helper = self._get_params_and_match_helper(name_match.name_type)
        orig_names, norm_names = match_helper.get_names([name_match.name_index])
        return set([on.name for on in orig_names if on.entity_id == name_match.entity_id]),\
               get_set_element(norm_names)

    # =====================================================================================================
    #       Methods for entity matching
    # =====================================================================================================

    def get_full_matches(self, text: str) -> List[NameMatchImpl]:
        """
        Entities matching entire text, i.e. entire text should match each name.
        Returned matches are sorted on:
            (name-tier [ascending], key_length [descending], start_token_index [asc.], entity_id [asc.]),
        ... where:
            key_length = Nbr. normalized tokens that matched (i.e. prefer longer matches)
            entity_id corresponds to sort order on Entity-IDs:
                in case of duplicated names, 'earlier' Entity-ID is preferred.
        """
        name_matches = set()
        match_helper = None
        for norm_type, match_helper in self.match_helpers.items():
            if match_helper.nbr_keys() > 0:
                tokens = self.tknzr.tokenize(text, normalization_type=norm_type)
                name_matches.update(match_helper.get_full_matches(tokens))

        # Use the last `match_helper` to get access to the `sorted_matches()` fn.
        return match_helper.sorted_matches(name_matches)

    def get_all_matching_entities(self, text: str,
                                  nmax: int = None) -> List[NameMatchImpl]:
        """
        All Entity matches into text, possibly overlapping.
        Returned matches are sorted on:
            (name-tier [ascending], key_length [descending], start_token_index [asc.], entity_id [asc.]),
        ... where:
            key_length = Nbr. normalized tokens that matched (i.e. prefer longer matches)
            entity_id corresponds to sort order on Entity-IDs:
                in case of duplicated names, 'earlier' Entity-ID is preferred.
        """
        name_matches = set()
        match_helper = None
        for norm_type, match_helper in self.match_helpers.items():
            if match_helper.nbr_keys() > 0:
                tokens = self.tknzr.tokenize(text, normalization_type=norm_type)
                name_matches.update(match_helper.get_all_sub_matches(tokens))

        # Use the last `match_helper` to get access to the `sorted_matches()` fn.
        matches = match_helper.sorted_matches(name_matches)
        if nmax:
            matches = matches[:nmax]

        return matches

    def get_greedy_nonoverlapping_matches(self, text: str,
                                          nmax: int = None) -> List[NameMatchImpl]:
        """
        Greedy selection of non-overlapping matches, prefers:
            - higher tiers
            - longer matching keys (in nbr of tokens)
            - left-most matches
            - 'earlier' entity_id (in sort order)
        """
        name_matches = self.get_all_matching_entities(text, nmax=nmax)

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
