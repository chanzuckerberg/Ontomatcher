"""
Misc utilities
"""

import contextlib
from datetime import datetime
import inspect
import itertools
import pickle
import sys
from typing import Any, Callable, Dict, Mapping, Sequence, Set, Tuple
import xml.etree.ElementTree as ET


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

DEFAULT = object()


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

        # prev_rec_limit = sys.getrecursionlimit()
        # sys.setrecursionlimit(32000)

        self._fpath = fpath

        with open(fpath, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        # sys.setrecursionlimit(prev_rec_limit)

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

    module = os.path.relpath(sys.argv[0], ".")
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


def pprint_xml(xml_elem: ET.Element,
               space='  ', level=0, is_last_child=True, file=sys.stdout):
    """
    pretty-print XML to stdout for tree rooted at `xml_elem`.
    Assumes SPACE is meaningless between tags, and between tag and text.
    So element.text and element.tail have SPACE stripped from both ends,
    and element.tail has SPACE then added at right for pretty-formatting.

    :param xml.etree.ElementTree.Element xml_elem: The tree rooted at this element will be printed.
    :param str space: The blank chars to use as single indent space at start of each line
    :param int level: Start indent level. leave this as default.
    :param is_last_child: for internal use, leave this as default.
    :param file: where to print
    """
    assert isinstance(level, int) and level >= 0

    indent = space * level

    print(indent, '<', xml_elem.tag, sep='', end='', file=file)
    if xml_elem.attrib:
        print(' ', ' '.join('{}="{}"'.format(k, v) for k, v in xml_elem.items()), sep='', end='', file=file)
    print('>', end='', file=file)

    my_text = xml_elem.text.strip() if xml_elem.text else None
    if my_text:
        print(my_text, sep='', end='', file=file)

    last_i = len(xml_elem) - 1
    for i, elem_ in enumerate(xml_elem):
        print(file=file)
        pprint_xml(elem_, space=space, level=level + 1, is_last_child=i == last_i, file=file)

    if len(xml_elem) > 0:
        print(indent, end='', file=file)
    print('</', xml_elem.tag, '>', sep='', end='', file=file)
    if is_last_child:
        print(file=file)

    my_tail = xml_elem.tail.strip() if xml_elem.tail else None
    if my_tail:
        print(indent, my_tail, sep='', end='\n', file=file)
        # if is_last_child:
        #     indent = space * (level - 1)
        #     print(indent, end='')

    return


@contextlib.contextmanager
def timed_exec(name: str, prefix: str = "-*- Time to", pre_msg: str = None, time_as_suffix_to_pre_msg=False, file=None):
    """
    Use this to print execution times of blocks.
    IF `time_as_suffix_to_pre_msg` AND `pre_msg` THEN
        run-time is added to end of line of `pre_msg`, and `name` is ignored

    >>> with timed_exec("running test", "Testing ..."):
    >>>     ...
    """
    if pre_msg is not None:
        print(pre_msg, file=file, end="" if time_as_suffix_to_pre_msg else "\n", flush=True)
    else:
        time_as_suffix_to_pre_msg = False

    t0 = datetime.now()
    yield
    if time_as_suffix_to_pre_msg:
        print("", datetime.now() - t0, file=file, flush=True)
    else:
        print(prefix, name, "=", datetime.now() - t0, file=file, flush=True)
    return


def dict_set_defaults(in_dict: Dict[str, Any], default_val: Any, key=None) -> Dict[str, Any]:
    """
    Update `in_dict`, setting default values from `default_val` if `in_dict` has no value (or None) for that key.
    Nested dictionaries also handled, as specified in `default_val`.

    :param in_dict: A Dict which will be updated

    :param default_val: Either a Dict or a Value.
        IF  `default_val` is a Dict THEN `key` must be None
        ELSE `key` must be not None

    :param key: IF key is not None THEN in_dict[Key] <- default value `default_val`

    :return: Updated in_dict.
    """
    assert isinstance(default_val, Mapping) or key is not None, "With no key arg, default_val must be a Mapping!"

    if key is not None:
        try:
            opt_val = in_dict.get(key, DEFAULT)
        except AttributeError:
            raise TypeError('Argument `in_dict` is not a Mapping!')

        if opt_val is DEFAULT:
            in_dict[key] = default_val
            return in_dict

    if isinstance(default_val, Mapping):
        if key is not None:
            in_dict = in_dict[key]
            assert isinstance(in_dict, Mapping), \
                f"Value of key '{key}' in arg `in_dict` must be a Mapping!"

        for subkey, subval in default_val.items():
            dict_set_defaults(in_dict, subval, subkey)

    return in_dict


def start_stop_index(seq: Sequence, keyfunc: Callable = None) -> Dict[Any, Tuple[int, int]]:
    """
    Returns {key: (start_idx, stop_idx)} for each unique key value in `seq`,
    s.t. all elements in seq[start_idx : stop_idx] have the key-val = key.
    Assumes that seq is sorted on key.

    :param seq: Sequence of elements, sorted on a key. Note that `None` should not be a valid key-value.
    :param keyfunc: Function that takes single argument, an element of seq, and returns the key-value.
        Default is identity function.
    :return: dict
    """
    if keyfunc is None:
        keyfunc = lambda x: x

    key_idx: Dict[Any, Tuple[int, int]] = dict()

    prev_key = None
    i, start_idx = 0, 0
    for i, element in enumerate(seq):
        key = keyfunc(element)
        if prev_key != key:
            if prev_key is not None:
                key_idx[prev_key] = (start_idx, i)
            start_idx = i
            prev_key = key

    key_idx[prev_key] = (start_idx, i + 1)
    return key_idx


def get_set_element(s: Set[Any]):
    """
    Get any element from the set, without removing it
    """
    return s.__iter__().__next__()


def firstnot(predicate, iterable):
    """First element in iterable that does not satisfy predicate."""
    try:
        return next(itertools.dropwhile(predicate, iterable))
    except StopIteration:
        return None


def list_without_nones(iterable):
    """List of all non-None elements"""
    return [e for e in iterable if e is not None]


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
