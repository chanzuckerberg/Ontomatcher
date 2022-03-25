"""
Build Ontology Matcher.
Find mentions of ontology terms in provided text, and in Napari plug-in descriptions.
"""

from collections import defaultdict
from itertools import chain
import json
import os
import os.path
import re
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

from ontomatch.data.ontology import Ontology
from ontomatch.data.imgontology import CuratedTerm, get_curated_imaging_subontology
from ontomatch.data.napari import fetch_plugin
from ontomatch.text.triematcher import NameMatch, TrieMatcher
from ontomatch.utils.misc import terminal_highlighted, highlight_spans_multicolor


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

URL_PATT = re.compile(r"\bhttp\S+")

INSTALLATION_HEADER = re.compile(r"^Installation$", flags=re.IGNORECASE)


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------

class EntityMatch(NamedTuple):
    entity_id: str
    primary_name: str

    # All the original (un-normalized) names that matcbed one or more mentions
    matching_names_original: Set[str]

    # All the normalized names that matched one or more mentions
    matching_names_normalized: Set[str]
# /


# -----------------------------------------------------------------------------
#   Functions: Ontology Matcher
# -----------------------------------------------------------------------------


def read_opts_chage_cwd(opts_json: str):
    """
    Read options from JSON file.
    IF JSON file is in a diff dir from cwd, THEN change cwd to that dir.
    This will allow paths mentioned in `opts_json` to be local to the dir containing `opts_json`.
    """
    with open(opts_json) as f:
        opts = json.load(f)

    cwd = os.getcwd()
    opts_dir = os.path.abspath(os.path.dirname(opts_json))
    if cwd != opts_dir:
        os.chdir(opts_dir)

    return opts, cwd


def build_imgont_trie_matcher(imgont_matcher_opts: Union[str, Dict[str, str]], verbose: bool = False) -> TrieMatcher:
    """
    Builds a Trie-based matcher for the Imaging sub-ontology with curated synonyms.
    Compiles and saves it.

    Required Keys:
        - EDAM: Path to EDAM ontology TSV file.
        - imgont: Path to imaging sub-ontology options (JSON), specifying subset of EDAM ontology.
        - curated_syns: Path to CSV file containing curated synonyms for imaging sub-ontology.
        - matcher: Path to JSON file containing TrieMatcher options.

    All relative paths are relative to the Python dir.
    """
    # noinspection PyUnresolvedReferences
    from ontomatch.text.triematcher import NameMatch, TrieMatcher

    cwd = None
    if isinstance(imgont_matcher_opts, str):
        imgont_matcher_opts, cwd = read_opts_chage_cwd(imgont_matcher_opts)

    trie_matcher = TrieMatcher.from_params(imgont_matcher_opts["matcher"])

    img_ont = get_curated_imaging_subontology(edam_ontology_tsv = imgont_matcher_opts["EDAM"],
                                              imgsubont_json = imgont_matcher_opts["imgont"],
                                              curated_syns_csv = imgont_matcher_opts["curated_syns"],
                                              verbose = verbose)

    # Heuritic expansion of Synonyms
    expand_ontology_synonyms(img_ont)

    # noinspection PyTypeChecker,PyUnusedLocal
    term: CuratedTerm = None

    for termid, term in img_ont.terms.items():
        if not term.ignore:
            trie_matcher.add_entity(entity_id = termid,
                                    primary_name = term.name,
                                    synonyms = term.curated_synonyms_full,
                                    acronyms = term.curated_acronyms,
                                    partial_names = term.curated_synonyms_partial)

    trie_matcher.compile_and_save()

    # Chage cwd back
    if cwd is not None:
        os.chdir(cwd)

    return trie_matcher


def get_imgont_trie_matcher(imgont_matcher_opts: Union[str, Dict[str, str]], verbose: bool = False) -> TrieMatcher:
    """
    Load from cached file, if exists, else build and save it.
    """
    cwd = None
    if isinstance(imgont_matcher_opts, str):
        imgont_matcher_opts, cwd = read_opts_chage_cwd(imgont_matcher_opts)

    trie_matcher = TrieMatcher.from_params(imgont_matcher_opts["matcher"])

    try:
        trie_matcher.load_from_cache()
    except FileNotFoundError:
        trie_matcher = build_imgont_trie_matcher(imgont_matcher_opts, verbose)

    # Chage cwd back
    if cwd is not None:
        os.chdir(cwd)

    return trie_matcher


# -----------------------------------------------------------------------------
#   Functions: Get matching terms
# -----------------------------------------------------------------------------


def get_ont_matches_by_line(text: str, ont_matcher: TrieMatcher,
                            htc_reduce_descr: bool = False) -> Tuple[List[str], List[List[NameMatch]]]:
    """
    Splits `text` into lines (on "\n"), and then looks for Ontology Term matches in each line.

    :param text: Text in which to find term mentions
    :param ont_matcher:
    :param htc_reduce_descr: Whether to heuristically reduce the text

    :return: Tuple:
        - List[ text-line ] as derived from `text`. IF `htc_reduce_descr` is True THEN this text may have been filtered.
        - List[ List of NameMatch, for each text-line ]
    """

    # `text` is messy HTML scrape. Split it into separate lines, so matches do not cross Sentences / Paras
    txt_lines = [line.strip() for line in text.splitlines()]
    if htc_reduce_descr:
        txt_lines = filter_descr_lines(txt_lines)

    matches_by_line = [ont_matcher.get_greedy_nonoverlapping_matches(line) for line in txt_lines]

    # Heuristically remove spurious matches
    matches_by_line = [heuristic_filter_matches(txt_line_, matches_)
                       for txt_line_, matches_ in zip(txt_lines, matches_by_line)]

    return txt_lines, matches_by_line


def get_entity_matches(text: str, ont_matcher: TrieMatcher,
                       htc_reduce_descr: bool = False) -> Dict[str, EntityMatch]:
    """
    Returns a Dict[ entity_id (str) => EntityMatch ], for each entity_id whose mention(s) detected in `text`.

    :param text: Text in which to find term mentions
    :param ont_matcher:
    :param htc_reduce_descr: Whether to heuristically reduce the text
    """
    txt_lines, matches_by_line = get_ont_matches_by_line(text, ont_matcher, htc_reduce_descr)
    matches = defaultdict(list)
    for line_matches in matches_by_line:
        for nm in line_matches:
            matches[nm.entity_id].append(nm)

    ent_matches = dict()
    for ent_id, name_matches in matches.items():
        primary_name = ont_matcher.get_primary_name(ent_id)
        matching_names_orig = set()
        matching_names_normlzd = set()
        for nm in name_matches:
            orig_names, normlzd_name = ont_matcher.get_matching_names(nm)
            matching_names_orig |= orig_names
            matching_names_normlzd.add(normlzd_name)

        ent_matches[ent_id] = EntityMatch(ent_id, primary_name,
                                          matching_names_original=matching_names_orig,
                                          matching_names_normalized=matching_names_normlzd)

    return ent_matches


def get_matching_entity_ids(text: str, ont_matcher: TrieMatcher,
                            htc_reduce_descr: bool = False) -> Set[str]:
    """
    Return the set of matching Term IDs.
    :param text: Text in which to find term mentions
    :param ont_matcher:
    :param htc_reduce_descr: Whether to heuristically reduce the text
    :return: Set of term Entity-IDs
    """
    txt_lines, matches_by_line = get_ont_matches_by_line(text, ont_matcher, htc_reduce_descr)
    entity_ids = set([nm.entity_id for line_matches in matches_by_line for nm in line_matches])
    return entity_ids


# -----------------------------------------------------------------------------
#   Functions: Heuristic expansion of synonyms
# -----------------------------------------------------------------------------


def expand_ontology_synonyms(img_ont: Ontology):
    # noinspection PyTypeChecker,PyUnusedLocal
    term: CuratedTerm = None

    for termid, term in img_ont.terms.items():
        if not term.ignore:
            if new_names := htc_expand(term.curated_synonyms_full | {term.name}):
                # print(f"Adding to {termid}:", " | ".join(new_names))
                term.curated_synonyms_full.update(new_names)

    return


def htc_expand(names: Sequence[str]) -> Set[str]:
    names = set(names)
    exp_names = set()

    for name_ in names:

        # '...oscopy' -> '...oscope'
        if name_.endswith("oscopy"):
            new_name = name_[:-1] + "e"
            if new_name not in names:
                exp_names.add(new_name)

    return exp_names


# -----------------------------------------------------------------------------
#   Functions: Heuristic filtering of matches
# -----------------------------------------------------------------------------

def heuristic_filter_matches(txt: str, matches: List[NameMatch]) -> List[NameMatch]:
    """
    Use heuristics to remove spurious matches
    """
    return remove_matches_overlapping_urls(txt, matches)


def get_url_spans(txt: str) -> List[Tuple[int, int]]:
    return [m.span(0) for m in URL_PATT.finditer(txt)]


def remove_matches_in_spans(name_matches: List[NameMatch], spans: List[Tuple[int, int]]) -> List[NameMatch]:
    def overlaps_prohibited_spans(s1, e1):
        nonlocal spans
        for s2, e2 in spans:
            if s1 < e2 and e1 > s2:
                return True
        return False

    if not spans:
        return name_matches
    else:
        return [nm for nm in name_matches if not overlaps_prohibited_spans(nm.char_start, nm.char_end)]


def remove_matches_overlapping_urls(txt: str, matches: List[NameMatch]) -> List[NameMatch]:
    """
    URLs in text should not be matched to any Entity, so remove any matches to those portions of the text.
    """
    url_spans = get_url_spans(txt)
    return remove_matches_in_spans(matches, url_spans)


# -----------------------------------------------------------------------------
#   Functions: Heuristic filtering of Plugin Description
# -----------------------------------------------------------------------------

def filter_descr_lines(txt_lines: List[str]):
    # Truncate on encountering header for "Installation"
    ti = 0
    for i, line in enumerate(txt_lines):
        if INSTALLATION_HEADER.match(line):
            ti = i
            break

    if ti > 0:
        return txt_lines[:ti]
    else:
        return txt_lines


# -----------------------------------------------------------------------------
#   Functions: Pretty-print matches
# -----------------------------------------------------------------------------


def pp_ont_matches(plugin_name: str, descr: str, ont_matcher: TrieMatcher,
                   htc_reduce_descr: bool = False):

    txt_lines, matches_by_line = get_ont_matches_by_line(descr, ont_matcher, htc_reduce_descr=htc_reduce_descr)

    reduced_descr = False
    if htc_reduce_descr:
        orig_n_lines = len(descr.splitlines())
        reduced_descr = orig_n_lines > len(txt_lines)

    matching_terms = defaultdict(set)
    for nmatch in chain.from_iterable(matches_by_line):
        matching_terms[nmatch.entity_id] |= ont_matcher.get_matching_names(nmatch)[0]

    print("Plugin:", terminal_highlighted(plugin_name, font_color='black'))
    print()
    for line, line_matches in zip(txt_lines, matches_by_line):
        matched_spans = set([(nmm.char_start, nmm.char_end) for nmm in line_matches])
        # noinspection PyTypeChecker
        print(highlight_spans_multicolor(line, {("blue", "bold"): matched_spans}))
        # print()

    if reduced_descr:
        print("... (Description heuristically reduced) ...")

    print()
    print(terminal_highlighted("Matched terms:", font_color='black'))

    if not matching_terms:
        print("... No term mentions found.")
        return

    for i, (termid, tmatches) in enumerate(sorted(matching_terms.items()), start=1):
        matched_names = sorted(tmatches)
        print(f"{i:2d}.  {termid}:", terminal_highlighted(ont_matcher.get_primary_name(termid)))
        print("       ", terminal_highlighted("Matches:", font_color='blue'), ", ".join(matched_names))

    return


def test_match_sample(imgont_matcher_opts: str, plugin_file: Optional[str] = None, plugin_name: Optional[str] = None,
                      htc_reduce_descr: bool = False):
    """
    Test ont matches on a single plugin
    :param imgont_matcher_opts: Path to JSON file.
    :param plugin_file: Path to text file containing Name on first line, and the rest is Description.
    :param plugin_name: Name of napari plugin, whose descr is fetched from Napari-Hub API.
    """
    plugin_descr: Optional[str] = None

    if plugin_file is not None:
        with open(plugin_file) as f:
            plugin_name = f.readline().strip()
            plugin_descr = f.read().strip()

    elif plugin_name is not None:
        print("Fetching data for", plugin_name, "...", flush=True)
        plugin_data = fetch_plugin(plugin_name)
        if plugin_data is None:
            print(f"Plugin with name {plugin_name} not found.")
            return

        plugin_descr = plugin_data.get_combined_description()

    ont_matcher = get_imgont_trie_matcher(imgont_matcher_opts)
    print()
    if htc_reduce_descr:
        print("Heuristically reduce description = True.")
        print()
    print("-------------------------------------------------------")
    pp_ont_matches(plugin_name, plugin_descr, ont_matcher, htc_reduce_descr=htc_reduce_descr)
    print("-------------------------------------------------------")
    return


# ======================================================================================================
#   Main
# ======================================================================================================

# Activate Virtual Env: $> . ~/Home/Test/PyVenvs/Txf46x/bin/activate
# Invoke as: python -m ontomatch.mentions.annotater CMD ...
# e.g.
# python -m ontomatch.mentions.annotater sample -f ../Data/plugin_sample.txt ../Data/imgont_matcher.json
# python -m ontomatch.mentions.annotater sample -n napari-allencell-segmenter ../Data/imgont_matcher.json

if __name__ == '__main__':

    import argparse
    from datetime import datetime
    from ontomatch.utils.misc import print_cmd

    _argparser = argparse.ArgumentParser(
        description='Bio-Image Processing Ontology Annotater.',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... build IMGONT_MATCHER_OPTS_JSON
    _sub_cmd_parser = _subparsers.add_parser('build',
                                             help="PPrint Samples.")

    _sub_cmd_parser.add_argument('imgont_matcher_opts_json', type=str,
                                 help="JSON file containing Image-Ontology matcher options.")

    # ... sample [-r] (-f PLUGIN_SAMPLE_FILE | -n PLUGIN_NAME) IMGONT_MATCHER_OPTS_JSON
    _sub_cmd_parser = _subparsers.add_parser('sample',
                                             help="PPrint Samples.")

    _group = _sub_cmd_parser.add_mutually_exclusive_group(required=True)

    _sub_cmd_parser.add_argument("-r", "--reduce_description", action='store_true',
                                 help="Use heuristics to reduce the description.")

    _group.add_argument("-f", "--plugin_sample_file", type=str,
                                 help="Sample file containing Plugin Name and Description.")

    _group.add_argument("-n", "--plugin_name", type=str,
                                 help="Plugin Name, as used in Napari Hub.")

    _sub_cmd_parser.add_argument('imgont_matcher_opts_json', type=str,
                                 help="JSON file containing Image-Ontology matcher options.")

    # ... annotate ONTOLOGY_CSV_FILE PLUGINS_CSV_FILE OUTPUT_CSV_FILE
    _sub_cmd_parser = _subparsers.add_parser('annotate',
                                             help="Annotate Plug-ins.")

    _sub_cmd_parser.add_argument('ontology_csv_file', type=str,
                                 help="CSV file containing Ontology Terms and Synonyms.")
    _sub_cmd_parser.add_argument('plugins_csv_file', type=str,
                                 help="CSV file containing Plugin Names and Descriptions.")
    _sub_cmd_parser.add_argument('output_csv_file', type=str,
                                 help="Output CSV file.")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print()
    print_cmd()

    if _args.subcmd == 'build':

        build_imgont_trie_matcher(_args.imgont_matcher_opts_json)

    elif _args.subcmd == 'sample':

        test_match_sample(_args.imgont_matcher_opts_json,
                          plugin_file=_args.plugin_sample_file,
                          plugin_name=_args.plugin_name,
                          htc_reduce_descr=_args.reduce_description)

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
