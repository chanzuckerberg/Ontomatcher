"""
High level API into EntityMatcher
"""

from abc import ABCMeta, abstractmethod
import dataclasses
from enum import Enum
import json
import os
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Union
import unicodedata

import regex

from utils.misc import PersistentObject, timed_exec


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class EntityNameType(Enum):
    # The preferred or primary name of an Entity
    PRIMARY = "primary"
    # Any alternative name that is not an Acronym
    SYNONYM = "synonym"
    # Any acronym that is not the primary name
    ACRONYM = "acronym"
    # Partial Name, lower priority synonyms, typically subsequences of Primary-Name or Synonyms
    PARTIAL = "partial"
# /


@dataclasses.dataclass
class EntityInfo:
    entity_id: str
    primary_name: str

    # For use by subclasses of `EntityMatcher`
    name_indices: Any = None
    # Typically used as {SYNONYM: (ss, se); ACRONYM: (as, ae)}
# /


class EntityMatcher(PersistentObject, metaclass=ABCMeta):
    """
    This Abstract Class' mainpurpose is to provide some common code, mostly for building and managing the data.
    The actual methods for matching are left to the sub-classes, as there are many different use cases based on
    the needs and implementation.

    To build an EntityMatcher:
        em = EntityMatcherSubclass.from_params(params_dict)
        em.add_entity(eid, e_primary, e_syns, e_acrs)
        ...
        em.compile_and_save()
    """

    def __init__(self, params: Dict[str, Any]):
        """
        params:
            - class: str.
                must match Class-Name
            - name: str.
                Some name for this set of options
            - descr: str.
                A brief description

            - cache_file: str.
                Path to file where Class Instance is cached by `build_and_save()`

            - lexicon_id: str.
                    Used to identify the Lexicon this matcher is built from.
                    Can be a path to the Lexicon file.

        """
        super().__init__()

        self.params = params
        self.name = params.get("name", self.__class__.__name__)
        self.descr = params.get("descr")
        self.lexicon_id = params["lexicon_id"]

        # eid -> {type_id: str, primary_name: str, primary_name_tokens: Array[str]}
        self.entity_info: Dict[str, EntityInfo] = dict()

        self.is_compiled = False

        return

    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate the loaded data's options as loaded from `self.load_data_cache()`
        against options specified in `__init__()` during creation of this instance.

        May potentially update `params`
        """
        raise NotImplementedError

    @abstractmethod
    def copy_from(self, another: "EntityMatcher"):
        """
        Copy data populated during `self.build_and_save()` from another instance.
        `another.params()` are equivalent (gives the same build) to this instance's params.

        Override, and call super().copy_from(another)
        """
        self.params = another.params
        self.name = another.name
        self.descr = another.descr
        self.lexicon_id = another.lexicon_id

        self.entity_info = another.entity_info

        self.is_compiled = another.entity_info
        return

    def load_from_cache(self):
        """
        Load data into `self.data` from option 'cache_file' and validate.
        """
        with timed_exec(name="Load " + self.__class__.__name__):
            loaded_obj = self.load_from(os.path.expanduser(self.params['cache_file']), verbose=True)

            print(f"  ... {self.__class__.__name__} Testing loaded object", flush=True)
            assert self.validate_params(loaded_obj.params), "Cached data params do not match!"

            print(f"  ... {self.__class__.__name__} Copying from loaded object", flush=True)
            self.copy_from(loaded_obj)

        return True

    def add_entity(self, entity_id: str, primary_name: str, synonyms: Sequence[str], acronyms: Sequence[str],
                   partial_names: Sequence[str] = None):
        assert not self.is_compiled, "Can only add entities before compiling"

        self.entity_info[entity_id] = EntityInfo(entity_id=entity_id, primary_name=primary_name)

        self._add_entity_names(entity_id, primary_name, synonyms, acronyms, partial_names)
        return

    @abstractmethod
    def _add_entity_names(self, entity_id: str,
                          primary_name: str, synonyms: Sequence[str], acronyms: Sequence[str],
                          partial_names: Sequence[str]):
        """
        Populate local data structures.
        """
        raise NotImplementedError

    @abstractmethod
    def compile_and_save(self):
        """
        Build out all the local data structures, and save to 'cache_file'
        """
        raise NotImplementedError

    @abstractmethod
    def get_normalized_name(self, name_index: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_original_name(self, name_index: int) -> str:
        raise NotImplementedError

    def get_original_names(self, name_index: int) -> Set[str]:
        """Override when returning multiple names mapping to same name-index."""
        return {self.get_original_name(name_index)}

    def get_all_entity_ids(self) -> Set[str]:
        return set(self.entity_info.keys())

    def get_primary_name(self, entity_id: str) -> str:
        return self.entity_info[entity_id].primary_name

    @abstractmethod
    def get_unique_synonyms(self, entity_id: str) -> Tuple[List[str], List[str]]:
        """
        Unique based on normalization scheme used by subclass.

        :return: Original-Synonyms, Normalized-Synonyms
            where first element corresponds to the primary name
        """
        raise NotImplementedError

    @abstractmethod
    def get_unique_acronyms(self, entity_id: str) -> Optional[Tuple[List[str], List[str]]]:
        """
        Unique based on normalization scheme used by subclass.

        :return: Original-Acronyms, Normalized-Acronyms
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
    #       Methods for entity matching are left to the sub-classes.
    # =====================================================================================================

# /


class Token(NamedTuple):
    text: str
    char_start: int
    char_end: int
# /


class BasicRevMappedTokenizer:
    """
    Equivalent to BasicTokenizer, except:
        - Maps each token to its character position in the source string.
        - Returns `List[Token]` instead of `List[str]`.
    """

    # Token consists of a sequence of:
    #   - Word-char (but not underscore '_')
    #   - Unicode Accent char e.g. [é] in 'Montréal' => 'Montreal'
    TOKEN_PATT = regex.compile(r"((?:(?!_)\w|\p{Mn})+)")

    def __init__(self):
        super().__init__()
        return

    def tokenize(self, txt: str, to_lower: bool = True) -> List[Token]:
        return self._tokenize(txt, to_lower=to_lower)

    def _tokenize(self, txt: str, to_lower: bool) -> List[Token]:
        tokens = []
        s = 0
        for t in self.TOKEN_PATT.split(txt):
            if len(t) == 0:
                continue
            elif not self.TOKEN_PATT.match(t):
                s += len(t)
            else:
                e = s + len(t)
                t = self.standardize_chars(t)
                if to_lower:
                    t = t.casefold()

                tokens.append(Token(t, s, e))
                s = e

        return tokens

    @staticmethod
    def standardize_chars(text):
        # Separate combined chars, e.g. [ﬁ] in 'ﬁnancial' => 'fi...'
        text = unicodedata.normalize("NFKD", text)

        # Strip accents, e.g. [é] in 'Montréal' => 'Montreal'
        text = "".join([c for c in text if unicodedata.category(c) != "Mn"])

        return text
# /
