"""
Misc utilities
"""

import inspect
import pickle
import sys
from typing import Any, Dict, Sequence, Set, Tuple


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------

class PersistentObject:
    def __init__(self):
        self._fpath = None
        super().__init__()

    @classmethod
    def load_from(cls, fpath, obj_name='', verbose=True):
        if verbose:
            if not obj_name:
                obj_name = cls.__name__
            print('Loading {} from: {} ...'.format(obj_name, fpath), end='', flush=True)

        with open(fpath, 'rb') as f:
            obj = pickle.load(f, fix_imports=False, encoding="UTF-8")

        obj._fpath = fpath

        if verbose:
            print(" completed.", flush=True)
        return obj

    def save(self, fpath, verbose=True):
        if verbose:
            print('Saving {} to: {} ...'.format(self.__class__.__name__, fpath), end='', flush=True)

        self._fpath = fpath

        with open(fpath, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        if verbose:
            print(flush=True)
        return
# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------

def print_cmd():
    import os
    import re

    module = os.path.relpath(sys.argv[0], "")
    module = module.replace("/", ".")
    module = re.sub(r"\.py$", "", module)
    print("$>", "python -m", module, *sys.argv[1:])
    print()
    return


def fn_name(fn):
    """Return str name of a function or method."""
    s = str(fn)
    if s.startswith('<function'):
        return 'fn:' + fn.__name__
    else:
        return ':'.join(s.split()[1:3])


def pp_funcargs(fn):
    arg_names = inspect.getfullargspec(fn).args
    print(fn_name(fn), "... args:")

    frame = inspect.currentframe()
    try:
        for i, name in enumerate(arg_names, start=1):
            if i == 1 and name == "self":
                continue
            print("   ", name, "=", frame.f_back.f_locals.get(name))

        print(flush=True)

    finally:
        # This is needed to ensure any reference cycles are deterministically removed as early as possible
        # see doc: https://docs.python.org/3/library/inspect.html#the-interpreter-stack
        del frame
    return


def get_set_element(s: Set[Any]):
    """
    Get any element from the set, without removing it
    """
    return s.__iter__().__next__()


# ======================================================================================================
#   Functions: Terminal Highlighting
# ======================================================================================================

ANSI_TERMINAL_FOREGROUND_COLORS = {
    'black':   '30',
    'red':     '31',
    'green':   '32',
    'yellow':  '33',
    'blue':    '34',
    'magenta': '35',
    'cyan':    '36',
    'white':   '97'
}

ANSI_TERMINAL_FORMATS = {
    'normal':     '0',
    'bold':       '1',
    'underlined': '4',
    'blinking':   '5',
    'reversed':   '7'
}


def terminal_highlighted(text, font_color='red', font_format='bold'):
    """Add control chars for highlighted printing on color terminal"""
    # return '\033[01;31m' + s + '\033[00m'
    clr_code = ANSI_TERMINAL_FOREGROUND_COLORS[font_color]
    fmt_code = ANSI_TERMINAL_FORMATS[font_format]
    return '\033[{};{}m{}\033[00m'.format(fmt_code, clr_code, text)


def highlight_spans(text, spans: Sequence[Tuple[int, int]], font_color='blue', font_format='bold'):
    txt_segments = []
    prev_ce = 0
    for cs, ce in sorted(spans):
        if cs > prev_ce:
            txt_segments.append(text[prev_ce : cs])
        txt_segments.append(terminal_highlighted(text[cs : ce], font_color, font_format))
        prev_ce = ce

    if prev_ce < len(text):
        txt_segments.append(text[prev_ce:])

    return "".join(txt_segments)


def highlight_spans_multicolor(text, format_spans_dict: Dict[Tuple[str, str], Sequence[Tuple[int, int]]]):
    """

    :param text:
    :param format_spans_dict:  (font_color, font_format) -> [(ch_start, ch_end), ...]
    :return:
    """
    # reverse dict
    spans_format = {tuple(span_): font_format
                    for font_format, span_seq in format_spans_dict.items() for span_ in span_seq}
    txt_segments = []
    prev_ce = 0
    for cs, ce in sorted(spans_format.keys()):
        if cs < prev_ce:
            continue
        elif cs > prev_ce:
            txt_segments.append(text[prev_ce : cs])

        font_color, font_format = spans_format[(cs, ce)]
        txt_segments.append(terminal_highlighted(text[cs : ce], font_color, font_format))
        prev_ce = ce

    if prev_ce < len(text):
        txt_segments.append(text[prev_ce:])

    return "".join(txt_segments)
