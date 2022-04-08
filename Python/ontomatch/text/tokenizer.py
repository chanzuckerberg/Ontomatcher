"""
A basic string tokenizer that provides mapping back to the source string.
"""

from enum import Enum, IntEnum, unique, auto
from typing import Any, Dict, List, NamedTuple
import unicodedata

import regex
from unidecode import unidecode

from nltk.stem import SnowballStemmer, WordNetLemmatizer


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------

class Token(NamedTuple):
    """
    Derived token with position in source string from which it was derived.
    Token is derived from `source[t.char_start, t.char_end]`.
    """

    # The derived token string.
    text: str

    # Start position in the source string from which token is derived.
    char_start: int

    # End position in the source string from which token is derived.
    char_end: int
# /


@unique
class NormalizationType(IntEnum):
    """Defines token normalization: any further processing of the derived token after Character Standardization."""

    # Ordinary tokenization, no change in case
    STANDARD = 0

    # All tokens are converted to Lower-Case
    LOWER = 1

    # All tokens are Stemmed (and Lower-Cased)
    STEMMED = 2

    # Lemmatized and lower-case
    LEMMATIZED = 3
# /


class CharacterStandardization(Enum):

    UNIDECODE = auto()
    """For `standardize_chars_unidecode()`"""

    BASIC = auto()
    """For `standardize_chars_basic()`"""

# /


class BasicRevMappedTokenizer:
    """
    Splits a source string into standardized and normalized tokens,
    and maps each token to its character position in the source string.
    """

    # Token consists of a sequence of:
    #   - Word-char (but not underscore '_')
    #   - Unicode Accent char e.g. [é] in 'Montréal' => 'Montreal'
    TOKEN_PATT = regex.compile(r"((?:(?!_)\w|\p{Mn})+)")

    DEFAULT_PARAMS = {
        # Character-standardization style
        "char_standardization": CharacterStandardization.UNIDECODE,
    }

    def __init__(self, params: Dict[str, Any] = None):

        self._set_params(params)

        self.use_unidecode_standardization = \
            self.params["char_standardization"] is CharacterStandardization.UNIDECODE

        # SnowballStemmer('english') is less aggressive than PorterStemmer
        self.stemmer = SnowballStemmer("english", ignore_stopwords=True)

        # For LEMMATIZED, more conservative than Stemming
        self.wnl = WordNetLemmatizer()

        return

    def _set_params(self, params):
        self.params = dict() if params is None else params

        if (char_standardization := self.params.get("char_standardization")) is None:
            self.params["char_standardization"] = self.DEFAULT_PARAMS["char_standardization"]
        elif not isinstance(char_standardization, CharacterStandardization):
            assert isinstance(char_standardization, str), f"Param 'char_standardization' must be a str"
            try:
                self.params["char_standardization"] = CharacterStandardization[char_standardization.upper()]
            except KeyError:
                raise KeyError(f"Illegal value 'char_standardization' = '{char_standardization}'")

        return

    def tokenize(self, txt: str, normalization_type: NormalizationType = NormalizationType.LOWER) -> List[Token]:
        return self._tokenize(txt, normalization_type=normalization_type)

    def normalize(self, txt: str, normalization_type: NormalizationType = NormalizationType.LOWER) -> str:
        return " ".join(self.normalized_tokens(txt, normalization_type=normalization_type))

    def normalized_tokens(self, txt: str, normalization_type: NormalizationType = NormalizationType.LOWER) -> List[str]:
        return [tkn.text for tkn in self.tokenize(txt, normalization_type=normalization_type)]

    def _tokenize(self, txt: str, normalization_type: NormalizationType) -> List[Token]:
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

                if normalization_type is NormalizationType.LOWER:
                    t = t.casefold()
                elif normalization_type is NormalizationType.STEMMED:
                    t = self.stemmer.stem(t)
                elif normalization_type is NormalizationType.LEMMATIZED:
                    t = self.wnl.lemmatize(t).casefold()

                tokens.append(Token(t, s, e))
                s = e

        return tokens

    def standardize_chars(self, text):
        if self.use_unidecode_standardization:
            return standardize_chars_unidecode(text)
        else:
            return standardize_chars_basic(text)
# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------

def standardize_chars_unidecode(text):
    # This seems to be a superset of `standardize_chars_basic`, and may be too aggressive for some use-cases.
    # E.g. this will convert 'μ-meter' to 'm-meter'
    text = unidecode(text)
    return text


def standardize_chars_basic(text):
    # Separate combined chars, e.g. [ﬁ] in 'ﬁnancial' => 'fi...'
    text = unicodedata.normalize("NFKD", text)

    # Strip accents, e.g. [é] in 'Montréal' => 'Montreal'
    text = "".join([c for c in text if unicodedata.category(c) != "Mn"])

    return text
