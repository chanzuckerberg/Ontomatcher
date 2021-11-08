"""
Basic NLP processing of text using stanza.
Ref: https://stanfordnlp.github.io/stanza/index.html
"""

from collections import defaultdict
import glob
import os.path
from typing import List, NamedTuple, Sequence

import stanza
import stanza.models.common.doc
#
# stanza.download('en')
#

from utils.misc import highlight_spans_multicolor, terminal_highlighted


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class EntitySpan(NamedTuple):
    text: str
    tokens: List[str]
    words: List[str]
    entity_type: str
    start_char: int
    end_char: int

    @classmethod
    def from_span(cls, span: stanza.models.common.doc.Span, char_offset: int = 0) -> "EntitySpan":
        # noinspection PyArgumentList
        return EntitySpan(text=span.text,
                          tokens=[t.text for t in span.tokens],
                          words=[w.text for w in span.words],
                          entity_type=span.type,
                          start_char=span.start_char - char_offset,
                          end_char=span.end_char - char_offset
                          )
# /


class NlpProcessor:
    """
    Tokenizes, and labels each token using BIOES notation.

    Entity types are (https://stanfordnlp.github.io/stanza/available_models.html#available-ner-models):

    4 Named Entity types from CoNLL03:

        PER, ORG, LOC, MISC

    18 Named Entity types from OntoNotes:

        PERSON:		People, including fictional
        NORP:		Nationalities or religious or political groups
        FACILITY:	Buildings, airports, highways, bridges, etc.
        ORG:	    (Organization) Companies, agencies, institutions, etc.
        GPE:		Countries, cities, states
        LOC:	    (Location) Non-GPE locations, mountain ranges, bodies of water
        PRODUCT:	Vehicles, weapons, foods, etc. (Not services)
        EVENT:		Named hurricanes, battles, wars, sports events, etc.
        WORK_OF_ART:	Titles of books, songs, etc.
        LAW:		Named documents made into laws
        LANGUAGE:	Any named language

        ... and Values, treated as Entities:

        DATE:		Absolute or relative dates or periods
        TIME:		Times smaller than a day
        PERCENT:	Percentage (including “%”)
        MONEY:		Monetary values, including unit
        QUANTITY:	Measurements, as of weight or distance
        ORDINAL:	“first”, “second”
        CARDINAL:	Numerals that do not fall under another type

    """

    # Entity Type colors dict
    ENT_COLORS = {
        "ORG":      ("blue", "bold"),
        "FACILITY": ("blue", "normal"),
        "PERSON":   ("cyan", "bold"),
        "GPE":      ("magenta", "bold"),
        "LOC":      ("magenta", "normal"),
        "NORP":     ("green", "bold"),
        "LANGUAGE": ("green", "normal"),
        "CARDINAL": ("red", "bold"),
        "QUANTITY": ("red", "normal"),
        "default":  ("black", "bold"),
    }

    # Entity Types
    ENT_ORGANIZATION = "ORG"
    ENT_LOCATION = "LOC"
    ENT_PERSON = "PER"

    def __init__(self):
        """
        """
        self.nlp = stanza.Pipeline('en', processors='tokenize,pos,ner')
        return

    def get_entity_mentions(self, txt: str, entity_types: Sequence[str] = None) -> List[EntitySpan]:
        doc = self.nlp(txt)
        org_mens = []
        for entspan in doc.entities:
            if entity_types and entspan.entity_type not in entity_types:
                continue

            org_mens.append(EntitySpan.from_span(entspan))

        return org_mens

    def get_entity_mentions_batched(self, txt_seq: List[str],
                                    entity_types: Sequence[str] = None) -> List[List[EntitySpan]]:
        """
        Returns all detected entity mentions of the requested `entity_types`.

        :param txt_seq: Sequence of str, each represents one 'document'
        :param entity_types: One of more of the entity types, e.g. ["ORG", "LOC"].
            Default is to return mentions of all entity types.

        :return: List[Doc_Entities]
            Doc_Entities = List[EntitySpan]
                            ... or empty if no Entity mentions in this document.
        """
        if txt_seq is None or len(txt_seq) == 0:
            return []

        if isinstance(txt_seq, str):
            txt_seq = [txt_seq]

        joined_txt, char_starts = self.join_txt_seq(txt_seq)
        doc = self.nlp(joined_txt)

        org_mens = [[] for _ in txt_seq]
        doc_idx = 0
        char_starts.append(len(joined_txt))

        for entspan in doc.entities:
            if entity_types and entspan.entity_type not in entity_types:
                continue

            # Assumes entities are in char sequence
            while entspan.start_char >= char_starts[doc_idx + 1]:
                doc_idx += 1

            org_mens[doc_idx].append(EntitySpan.from_span(entspan, char_offset=char_starts[doc_idx]))

        return org_mens

    @staticmethod
    def join_txt_seq(txt_seq):
        char_starts = [0]
        joined_txt = txt_seq[0]
        for txt in txt_seq[1:]:
            if not txt:
                continue
            joined_txt += '\n\n'
            char_starts.append(len(joined_txt))
            joined_txt += txt
        return joined_txt, char_starts

    def show_entities(self, txt: str):
        print()
        print(txt)
        print(flush=True)
        doc = self.nlp(txt)
        print(*[f'entity: {ent.text}\ttype: {ent.entity_type}' for sent in doc.sentences for ent in sent.ents], sep='\n')
        print(flush=True)
        return

    def pp_entity_mentions(self, txt: str):
        doc = self.nlp(txt)
        self.pp_entities(txt, doc.entities)
        return

    def pp_entities(self, txt: str, entities):
        if len(entities) == 0:
            print("   None.")
            return

        ent_spans = defaultdict(list)
        ent_colors_key = defaultdict(set)
        for ent in entities:
            color = self.ENT_COLORS.get(ent.type, self.ENT_COLORS["default"])
            ent_colors_key[color].add(ent.type)
            ent_spans[color].append((ent.start_char, ent.end_char))

        print("Entity mentions:")
        print(highlight_spans_multicolor(txt, ent_spans))

        print()
        print("KEY:")

        maxw = max(len(", ".join(v)) for v in self.ENT_COLORS.values()) + 12

        for color in sorted(ent_colors_key.keys()):
            ent_types = ', '.join(ent_colors_key[color])
            print("    {:{w}s}:  {:s}".format(
                terminal_highlighted(', '.join(color), font_color=color[0], font_format=color[1]),
                terminal_highlighted(ent_types, font_color=color[0], font_format=color[1]),
                w=maxw
            ))

        return
# /
