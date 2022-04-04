"""
High level API into EntityMatcher
"""

from abc import ABCMeta, abstractmethod
from collections import defaultdict
import dataclasses
import json
import os
from typing import Any, Dict, List, Sequence, Set, Tuple, Union

from .tokenizer import NormalizationType

from ontomatch.utils.misc import PersistentObject


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class EntityInfo:
    entity_id: str
    primary_name: str

    # For use by subclasses of `EntityMatcher`: NameType => Set[indices]
    name_indices: Dict[str, Set[int]] = dataclasses.field(default_factory=lambda: defaultdict(set))
# /


@dataclasses.dataclass(frozen=True)
class NameMatch:
    """
    Instances of this (or its sub-class) are returned for each name match.
    Setting `frozen=True` makes instance immutable and hashable, so can be used in a Set.
    """

    entity_id: str
    name_type: str

    # Identifies which name matched.
    # Needed to retrieve original and normalized names that resulted in this match
    name_index: int

    # Position of match in source string. Match is at `src[char_start : char_end]`.
    char_start: int
    char_end: int
# /


class EntityMatcher(PersistentObject, metaclass=ABCMeta):
    """
    This Abstract Class' main purpose is to provide some common code, mostly for building and managing the data.
    The actual methods for matching are left to the sub-classes, as there are many different use cases based on
    the needs and implementation.

    To build an EntityMatcher:
        em = EntityMatcherSubclass.from_params(params_dict)
        em.add_entity(eid, e_primary, e_syns, e_acrs)
        ...
        em.compile_and_save()
    """

    EXAMPLE_PARAMS = {
        # Required. Specifies sub-class of EntityMatcher to create.
        "class": "TrieMatcher",

        # Give it a meaningful name
        "name": "Default Matcher",

        # Brief description
        "descr": "Matcher with default parameters",

        # Identifies where the entities and their names are from
        "lexicon_id": "EDAM Imaging sub-Ontology",

        # Any single-token Entity Names whose length is less than this value will be ignored.
        # Default is 2.
        "min_name_length": 2,

        # Entity names that match any of these (after corresponding normalization) are ignored.
        # Default is no stop-names.
        # Example:
        #   Based on the "name_types" below, an entry of "Ms" will result in
        #       - "primary", "synonym" and "partial" names that are normalized to "ms" being ignored
        #       - "acronym" name "Ms" will be ignored, but not "MS" or "ms".
        "stop_names": ["as", "The"],

        # How to match the different types of names:
        #  NameType [str] => [Dict]
        #                    {"tier": [int > 0] Match tier,
        #                     "normalization": [str] A name of one of the members of `tokenizer.NormalizationType`
        #                    }
        "name_types": {
            # An entry for "primary" is required, but the value can be customized.
            # This NameType is used for each entity's Primary-Name.
            "primary": {"tier": 1, "normalization": "LOWER"},

            # Other NameTypes ...
            "acronym": {"tier": 2, "normalization": "STANDARD"},
            "synonym": {"tier": 3, "normalization": "LOWER"},
            "partial": {"tier": 4, "normalization": "LOWER"},
        }
    }

    def __init__(self, params: Dict[str, Any]):
        """
        Params
        ------

        name: str.
            Some name for this set of options
        descr: str.
            A brief description

        cache_file: str.
            Path to file where Class Instance is cached by `build_and_save()`, as a Pickle file.

        lexicon_id: str.
                Used to identify the Lexicon this matcher is built from.
                Can be a path to the Lexicon file.

        name_types: Dict[ NameType [str] => Dict[ NameType-Definition ]
            Defines how to normalize names of this type, and their match tier.
            See `EXAMPLE_PARAMS`.

        min_name_length: int, default = 2
                For single-token names, their length (# chars) should be at least this much,
                else they are discarded.

        stop_names: List[str]
            e.g. ["as"]
        """
        super().__init__()

        self.params = params
        self.name = params.get("name", self.__class__.__name__)
        self.descr = params.get("descr")
        self.lexicon_id = params.get("lexicon_id")

        self.name_type_params = params["name_types"]
        self._validate_name_type_params()

        self.min_name_length = params.get("min_name_length", 2)
        self.stop_names = params.get("stop_names", [])

        self.cache_file = params.get("cache_file")

        # Local fields

        self.is_compiled = False

        # eid -> {type_id: str, primary_name: str, primary_name_tokens: Array[str]}
        self.entity_info: Dict[str, EntityInfo] = dict()

        return

    def _validate_name_type_params(self) -> bool:
        """
        Checks all name-type definitions, and converts 'normalization' values to `NormalizationType` members.
        """
        assert "primary" in self.name_type_params, f"'primary' is a required key in `name_type_params`"

        for nt, np in self.name_type_params.items():
            assert "tier" in np and "normalization" in np, \
                f"Illegal value of name_type_params['{nt}']: 'tier' and 'normalization' are required keys."

            tier = np["tier"]
            assert isinstance(tier, int) and tier > 0, f"{nt}.tier = {tier} should be int > 0"

            norm = np["normalization"]
            if not isinstance(norm, NormalizationType):
                assert isinstance(norm, str), \
                    f"name_type_params['{nt}']['normalization'] must be str or `NormalizationType`"
                try:
                    np["normalization"] = NormalizationType[norm.upper()]
                except KeyError:
                    raise KeyError(f"Illegal value name_type_params['{nt}']['normalization'] = '{norm}'")

        return True

    def matches_loaded_obj(self, loaded_obj: "EntityMatcher") -> bool:
        """
        Ensure loaded object's options match self.
        """
        return self.__class__ is loaded_obj.__class__ and \
               self.name_type_params == loaded_obj.name_type_params

    @abstractmethod
    def copy_from(self, another: "EntityMatcher"):
        """
        Copy data populated during `self.build_and_save()` from another instance.
        `another.params()` are equivalent (gives the same build) to this instance's params.

        Override, and call super().copy_from(another)
        """
        self.name = another.name
        self.descr = another.descr
        self.lexicon_id = another.lexicon_id
        self.name_type_params = another.name_type_params

        self.entity_info = another.entity_info

        self.is_compiled = another.entity_info
        return

    def load_from_cache(self):
        """
        Load data into `self.data` from option 'cache_file' and validate.
        """
        loaded_obj = self.load_from(os.path.expanduser(self.cache_file), verbose=True)

        print(f"  ... {self.__class__.__name__} Testing loaded object", flush=True)
        assert self.matches_loaded_obj(loaded_obj), "Cached data params do not match!"

        print(f"  ... {self.__class__.__name__} Copying from loaded object", flush=True)
        self.copy_from(loaded_obj)

        return True

    def add_entity(self, entity_id: str, primary_name: str, synonyms: Sequence[str], acronyms: Sequence[str],
                   partial_names: Sequence[str] = None):
        """
        Convenience method, using the name-types as defined in `EXAMPLE_PARAMS`
        """
        self.add_entity_with_names(entity_id, primary_name,
                                   dict(synonym=synonyms, acronym=acronyms, partial=partial_names))
        return

    def add_entity_with_names(self, entity_id: str, primary_name: str,
                              typed_synonyms: Dict[str, Sequence[str]]):

        assert not self.is_compiled, "Can only add entities before compiling"

        self.entity_info[entity_id] = EntityInfo(entity_id=entity_id, primary_name=primary_name)

        self._add_names_for_entity(entity_id, "primary", [primary_name])

        for name_type, names in typed_synonyms.items():
            if not names:
                continue

            self._add_names_for_entity(entity_id, name_type, names)

        return

    @abstractmethod
    def _add_names_for_entity(self, entity_id: str, name_type: str, names: Sequence[str]):
        """
        Add alternative names for existing entity.
        Populate local data structures.
        """
        raise NotImplementedError

    @abstractmethod
    def compile_and_save(self):
        """
        Build out all the local data structures, and save to 'cache_file'
        """
        raise NotImplementedError

    def get_all_entity_ids(self) -> Set[str]:
        return set(self.entity_info.keys())

    def get_primary_name(self, entity_id: str) -> str:
        return self.entity_info[entity_id].primary_name

    @abstractmethod
    def get_all_unique_names(self, entity_id: str) -> Dict[str, Tuple[Set[str], Set[str]]]:
        """
        :return: Dict: NameType[str] => Tuple[ Set[Orig Name], Set[Normalized Name] ]
            ... where NameType's are as defined in `self.name_type_params`.
        """
        raise NotImplementedError

    @abstractmethod
    def get_unique_names(self, entity_id: str, name_type: str) -> Tuple[Set[str], Set[str]]:
        """
        :return: For specified `entity_id` and `name_type`,
            Set[ Original-Names (un-normalized) ], Set[ Normalized-Names ]
        """
        raise NotImplementedError

    # @abstractmethod
    # def get_original_names(self, entity_id: str, name_type: str, name_index: int) -> Sequence[str]:
    #     """
    #     Get the original (un-normalized) names associated with `entity_id`, `name_type` and `name_index`.
    #     Used to retrieve the original names related to a `NameMatch`.
    #     """
    #     raise NotImplementedError

    @abstractmethod
    def get_matching_names(self, name_match: NameMatch) -> Tuple[Set[str], str]:
        """
        Get the original (un-normalized) names and the single normalized name associated with a `NameMatch`.
        :return: Set[ Original-Names (un-normalized) ], Normalized-Name
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Union[str, Dict[str, Any]]):
        """
        :param params: Either params dict, or path to JSON file
        """
        if isinstance(params, str):
            params_file = os.path.expanduser(params)
            with open(params_file) as f:
                params = json.load(f)

        assert params["class"] == cls.__name__
        return cls(params)

    # =====================================================================================================
    #       Methods for entity matching
    # =====================================================================================================

    @abstractmethod
    def get_full_matches(self, text: str) -> List[NameMatch]:
        """
        Entities matching entire text, i.e. entire text should match each name.
        Returned matches are sorted on (name-tier [ascending], key_length [descending]),
        ... where key_length = Nbr. normalized tokens that matched (i.e. prefer longer matches)
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_matching_entities(self, text: str,
                                  nmax: int = None) -> List[NameMatch]:
        """
        All Entity matches into text, possibly overlapping.
        Returned matches are sorted on (name-tier [ascending], key_length [descending]),
        ... where key_length = Nbr. normalized tokens that matched (i.e. prefer longer matches)
        """
        raise NotImplementedError

    @abstractmethod
    def get_greedy_nonoverlapping_matches(self, text: str,
                                          nmax: int = None) -> List[NameMatch]:
        """
        Greedy selection of non-overlapping matches, prefers:
            - higher tiers
            - longer matching keys (in nbr of tokens)
            - left-most matches
        """
        raise NotImplementedError

# /
