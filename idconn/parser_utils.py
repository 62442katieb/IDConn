"""Utility functions for IDconn argument parsers."""
import os.path as op


def is_valid_path(parser, arg):
    """Check if argument is existing folder."""
    if not op.isdir(arg) and arg is not None:
        parser.error(f'The folder {arg} does not exist!')

    return arg
