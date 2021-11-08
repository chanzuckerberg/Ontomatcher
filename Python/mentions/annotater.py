"""
Find mentions of ontology terms in Napari plug-in descriptions
"""

import csv
from collections import defaultdict
from itertools import chain
import re
from typing import List, Tuple

from data.datacsv import read_ontology, read_plugins
from text.triematcher import NameMatch, TrieMatcher
from utils.misc import terminal_highlighted, highlight_spans_multicolor


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

URL_PATT = re.compile(r"\bhttp\S+")


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def build_ontology_trie_matcher(ontology_csv_file: str):
    """
    """
    trie_matcher = TrieMatcher()

    ontology = read_ontology(ontology_csv_file)

    for termid, tval in ontology.items():
        acronyms = set([name for name in tval["synonyms"] | tval["partials"] if name.isupper()])
        synonyms = tval["synonyms"] - acronyms
        partials = tval["partials"] - acronyms

        # noinspection PyTypeChecker
        trie_matcher.add_entity(termid, tval["name"], synonyms, acronyms, partials)

    return trie_matcher


def get_ont_matches(text: str, ont_matcher: TrieMatcher) -> Tuple[List[str], List[List[NameMatch]]]:
    # `text` is messy HTML scrape. Split it into separate lines, so matches do not cross Sentences / Paras
    txt_lines = [line.strip() for line in text.splitlines() if line.strip()]
    matches = [ont_matcher.get_greedy_nonoverlapping_matches(line) for line in txt_lines]
    return txt_lines, matches


def pp_ont_matches(plugin_name: str, descr: str, ont_matcher: TrieMatcher):
    txt_lines, matches_by_line = get_ont_matches(descr, ont_matcher)

    # Remove URLs, in case they cause spurious matches
    txt_lines = [re.sub(URL_PATT, "", line) for line in txt_lines]

    matching_terms = defaultdict(set)
    for nmatch in chain.from_iterable(matches_by_line):
        matching_terms[nmatch.entity_id] |= ont_matcher.get_original_names(nmatch.name_index)

    print("Plugin:", terminal_highlighted(plugin_name, font_color='black'))
    print()
    for line, line_matches in zip(txt_lines, matches_by_line):
        matched_spans = set([(nmm.char_start, nmm.char_end) for nmm in line_matches])
        # noinspection PyTypeChecker
        print(highlight_spans_multicolor(line, {("blue", "bold"): matched_spans}))
        # print()

    print()
    print(terminal_highlighted("Matched terms:", font_color='black'))
    for i, (tname, tmatches) in enumerate(sorted(matching_terms.items()), start=1):
        matched_names = sorted(tmatches)
        print(f"{i:2d}.  {tname}")
        print("       ", terminal_highlighted("Matches:", font_color='blue'), ", ".join(matched_names))

    return


def pp_plugin_matching_terms(ontology_csv_file: str, plugins_csv_file: str, maxcount: int = 0):
    ont_matcher = build_ontology_trie_matcher(ontology_csv_file)

    pn = 0
    for plugin in read_plugins(plugins_csv_file):
        pn += 1
        print(f"----- [{pn}] --------------------------------------------------------------------------------------")
        print()
        descr = plugin.description
        if plugin.summary:
            descr = plugin.summary + "\n" + descr
        pp_ont_matches(plugin.name, descr, ont_matcher)
        print()

        if pn == maxcount > 0:
            break

    print("=======================================================================================================")
    print(f"{pn} Plugins annotated.")
    return


def annotate_plugins(ontology_csv_file: str, plugins_csv_file: str, output_csv_file: str):
    ont_matcher = build_ontology_trie_matcher(ontology_csv_file)

    with open(output_csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, dialect='excel')
        # Col Header
        csvwriter.writerow(["PluginNbr", "Name", "Terms", "TermMatches"])

        pn = 0
        for plugin in read_plugins(plugins_csv_file):
            pn += 1

            descr = plugin.description
            if plugin.summary:
                descr = plugin.summary + "\n" + descr

            txt_lines, matches_by_line = get_ont_matches(descr, ont_matcher)

            matching_terms = defaultdict(set)
            for nmatch in chain.from_iterable(matches_by_line):
                matching_terms[nmatch.entity_id] |= ont_matcher.get_original_names(nmatch.name_index)

            found_terms = "\n".join(sorted(matching_terms.keys()))
            matches = "\n".join([", ".join(v) for k, v in sorted(matching_terms.items())])

            csvwriter.writerow([pn, plugin.name, found_terms, matches])

    print(f"{pn} Plugins annotated.")
    print("Output written to:", output_csv_file)
    return


# ======================================================================================================
#   Main
# ======================================================================================================

# Activate Virtual Env: $> . ~/Home/Test/PyVenvs/Txf46x/bin/activate
# Invoke as: python -m mentions.annotater CMD ...
# e.g.
# python -m mentions.annotater sample \
#   ../Data/programmatic\ napari-hub\ tagging\ -\ full\ terms\,synonyms\,\ partial\ phrases.csv
#   ../Data/programmatic\ napari-hub\ tagging\ -\ 10\ plugins.csv

if __name__ == '__main__':

    import argparse
    from datetime import datetime
    from utils.misc import print_cmd

    _argparser = argparse.ArgumentParser(
        description='Annotate Napari Plugins from Image Processing Ontology.',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... sample ONTOLOGY_CSV_FILE PLUGINS_CSV_FILE MAXCOUNT
    _sub_cmd_parser = _subparsers.add_parser('sample',
                                             help="PPrint Samples.")

    _sub_cmd_parser.add_argument('ontology_csv_file', type=str,
                                 help="CSV file containing Ontology Terms and Synonyms.")
    _sub_cmd_parser.add_argument('plugins_csv_file', type=str,
                                 help="CSV file containing Plugin Names and Descriptions.")
    _sub_cmd_parser.add_argument('maxcount', type=int, nargs='?', default=0,
                                 help="Max nbr Plugins to annotate.")

    # ... annotate ONTOLOGY_CSV_FILE PLUGINS_CSV_FILE OUTPUT_CSV_FILE
    _sub_cmd_parser = _subparsers.add_parser('annotate',
                                             help="Annotate Plug-ins.")

    _sub_cmd_parser.add_argument('ontology_csv_file', type=str,
                                 help="CSV file containing Ontology Terms and Synonyms.")
    _sub_cmd_parser.add_argument('plugins_csv_file', type=str,
                                 help="CSV file containing Plugin Names and Descriptions.")
    _sub_cmd_parser.add_argument('output_csv_file', type=str,
                                 help="Output CSV file.")

    # ... test
    _sub_cmd_parser = _subparsers.add_parser('test', help="Testing.")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print()
    print_cmd()

    if _args.subcmd == 'sample':

        pp_plugin_matching_terms(_args.ontology_csv_file, _args.plugins_csv_file,
                                 maxcount=_args.maxcount)

    elif _args.subcmd == 'annotate':

        annotate_plugins(_args.ontology_csv_file, _args.plugins_csv_file, _args.output_csv_file)

    elif _args.subcmd == 'test':

        print("Testing!")

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
