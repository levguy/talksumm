import os
import sys
from datetime import datetime
from scipy import spatial
import math
import textwrap


def tprint(str):
    print('[{0}] {1}'.format(datetime.now().strftime('%d.%m|%H:%M:%S'), str))
    sys.stdout.flush()
    return


def cosine_similarity(u, v):
    cosine_distance = spatial.distance.cosine(u, v)
    assert (not math.isnan(cosine_distance))
    cosine_sim = 1 - cosine_distance
    return cosine_sim


def files_in_dir(dir_name, sort=True):
    """
    returns a list of the files in a given directory
    """
    file_names = [fname for fname in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, fname))]
    if sort:
        file_names.sort()
    return file_names


# based on:
# https://stackoverflow.com/questions/17330139/python-printing-a-dictionary-as-a-horizontal-table-with-headers
def print_table(rows, col_names=None, sep='\uFFFA', num_fraction_digits=None, max_col_width=30, print_en=True):
    """
    Pretty print a list of dictionaries (rows) as a dynamically sized table.
    If column names (col_names) aren't specified, they will show in random order.
    The function prepares and returns a string representing the table, and optionally prints it, depending on print_en
    sep: row separator. Ex: sep='\n' on Linux. Default: dummy to not split line.
    num_fraction_digits: number of fraction digits to be printed, in case of float (set to None for printing all digits)
    example:
    print_table([{'a': 123, 'bigtitle': 456, 'c': 0.0, 'split\ntitle': 7},
                 {'a': 'x', 'bigtitle': 'y', 'c': 'long text to be split', 'split\ntitle': 8},
                 {'a': '2016-11-02', 'bigtitle': 1.23, 'c': 7891231, 'split\ntitle': 9}],
                ['a', 'bigtitle', 'c', 'split\ntitle'],
                sep='\n',
                num_fraction_digits=1,
                max_col_width=10)
    """
    if not col_names:
        col_names = list(rows[0].keys() if rows else [])
    my_list = [col_names]  # 1st row = header
    for item in rows:
        if num_fraction_digits is not None:
            for key in item.keys():
                if isinstance(item[key], float):
                    format_str = '{:.' + str(num_fraction_digits) + 'f}'
                    item[key] = format_str.format(item[key])

        my_list.append([sep.join(textwrap.wrap(str(item[col]), max_col_width)) for col in col_names])
    col_size = [max(map(len, (sep.join(col)).split(sep))) for col in zip(*my_list)]
    format_str = ' | '.join(["{{:<{}}}".format(i) for i in col_size])
    dash_line = format_str.replace(' | ', '-+-').format(*['-' * i for i in col_size])
    item = my_list.pop(0)
    dash_line_done = False
    lines = []
    while my_list:
        if all(not i for i in item):
            item = my_list.pop(0)
            if dash_line and (sep != '\uFFFA' or not dash_line_done):
                lines.append(dash_line)
                dash_line_done = True

        while any(item):
            row = [i.split(sep, 1) for i in item]
            line = format_str.format(*[i[0] for i in row])
            lines.append(line)
            item = [i[1] if len(i) > 1 else '' for i in row]

    out_str = '\n'.join(lines)
    if print_en:
        print(out_str)
    return out_str
