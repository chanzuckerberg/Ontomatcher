"""
For simple pretty-printed tables.
None of the available packages seemed good enough or simple enough, e.g. Tabulate, Pandas, Asciitable.

Not yet supported: Fancy formats like HTML, LaTeX.
"""

import numpy as np

from .misc import firstnot


class PrettyTable:
    """
    Usage::

        from pubmed.utils.prettytable import PrettyTable

        ptbl = PrettyTable()
        ptbl.add_column('name 1', ['v1', 'v2', ...])
        ptbl.add_column('name 2', [..., ints, ...], format_spec=',d')
        ptbl.add_column('name 2', [..., floats, ...], format_spec='.2f')
        print(ptbl)

    Not an efficient implementation -- not intended for large tables.
    """
    def __init__(self, sep='  '):
        self.hdrs = []
        self.cols = []
        self.formats = []
        self.isnumeric = []

        self.underline_char = '-'
        self.colspace = sep
        return

    def add_column(self, name, values, format_spec=None):
        """
        Add a column named `name` with `values`.

        :param str name: Column name.
        :param list, np.ndarray values: Sequence. Homogenous type values, can contain None.
        :param str format_spec: Format spec, e.g. '.3f'
        """
        assert isinstance(name, str)
        assert isinstance(values, (tuple, list)) or (isinstance(values, np.ndarray) and values.ndim == 1)

        if isinstance(values, np.ndarray):
            values = values.tolist()

        self.hdrs.append(name)
        self.cols.append(values)
        self.formats.append(format_spec)

        self.isnumeric.append(self._isnumeric(len(self.cols) - 1))
        return

    def set_colnames(self, names, format_specs=None):
        """
        Set col names to `names`.
        :param names: List or Tuple of str.
        :param format_specs: List or Tuple of str.
        """
        assert isinstance(names, (tuple, list))
        assert format_specs is None or isinstance(format_specs, (tuple, list))
        self.hdrs = names
        self.formats = format_specs or list()
        return

    def add_row_(self, *values):
        return self.add_row(values)

    def add_row(self, values):
        assert isinstance(values, (tuple, list))
        self._normalize_shape(len(values))
        for c, v in zip(self.cols, values):
            c.append(v)
        return

    def _normalize_shape(self, ncols):
        curr_ncols = len(self.cols)
        new_ncols = max(len(self.hdrs), curr_ncols, ncols)

        self.hdrs += ['col_' + str(i + 1) for i in range(len(self.hdrs), new_ncols)]

        if self.cols:
            nrows = max(len(c) for c in self.cols)
            self.cols = [c + [None] * (nrows - len(c)) for c in self.cols] + \
                        [[None for _ in range(nrows)] for _ in range(curr_ncols, new_ncols)]
        else:
            nrows = 0
            self.cols = [[] for _ in range(new_ncols)]
        return new_ncols, nrows

    def _normalize_formats(self, ncols=0):
        if not ncols:
            ncols, _ = self._normalize_shape(0)

        self.formats += list([None]) * (ncols - len(self.formats))
        self.isnumeric += [self._isnumeric(ci) for ci in range(len(self.isnumeric), ncols)]
        return

    def _isnumeric(self, ci):
        v = firstnot(lambda v_: v_ is None, self.cols[ci])   # Returns None if none found
        return np.isreal(v) if v is not None else False     # np.isreal(None) = True !

    def _pp_val(self, ci, v):
        if v is None:
            return ''
        fspec = self.formats[ci]
        if not fspec:
            fspec = '{}'
        else:
            fspec = '{:' + fspec + '}'
        return str.format(fspec, v)

    def _pp_valw(self, ci, ri, width=0):
        if ri == -2:
            vstr = self.hdrs[ci]
        elif ri == -1:
            vstr = self.underline_char * width
        else:
            vstr = '' if ri >= len(self.cols[ci]) else self._pp_val(ci, self.cols[ci][ri])
        if self.isnumeric[ci] or self.formats[ci] == ">s":
            return vstr.rjust(width)
        else:
            return vstr.ljust(width)

    def __str__(self):
        if not self.hdrs:
            return ''

        self._normalize_formats()

        width = len(self.hdrs)
        height = max(len(c) for c in self.cols)

        colws = [max(len(self._pp_valw(ci, ri)) for ri in range(-2, len(self.cols[ci]))) for ci in range(width)]

        return '\n'.join(self.colspace.join(self._pp_valw(ci, ri, colws[ci]) for ci in range(width))
                         for ri in range(-2, height)) + '\n'

    def print(self, tsv_mode=False, file=None):
        if tsv_mode:
            width = len(self.hdrs)
            height = max(len(c) for c in self.cols)
            print(*['\t'.join(self._pp_valw(ci, ri) for ci in range(width)) for ri in range(-2, height)],
                  sep='\n', file=file)
        else:
            print(self, file=file)
# /


# ======================================================================================================
#   Main
# ======================================================================================================

# To test, invoke as: python3 -m pubmed.utils.prettytable

if __name__ == '__main__':

    ptbl = PrettyTable()
    ptbl.add_column('Col 1', list(range(5)))
    ptbl.add_column('Col 2', list(10 + v/10 for v in range(5)), '.1f')
    print('2 cols x 5 rows:', ptbl, sep='\n')
    ptbl.add_row([6.1, 6.2, 6.3, 6.4])
    print('Added 2 empty cols, then 1 row:', ptbl, sep='\n')
    print()
    print("TSV Mode:")
    ptbl.print(tsv_mode=True)
