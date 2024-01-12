"""
IDConn: Individual Differences in brain Connectivity
"""
import warnings
import logging

from ._version import get_versions

logging.basicConfig(level=logging.INFO)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("ignore")
    from . import connectivity
    from . import data
    from . import nbs
    from . import networking

    # from . import preprocessing
    # from . import statistics
    # from . import utils
    from . import io

    __version__ = get_versions()["version"]

    __all__ = [
        "idconn",
        "connectivity",
        "data",
        # "figures",
        "networking",
        # "preprocessing",
        # "statistics",
        # "utils",
        "io",
        "nbs",
        "__version__",
    ]

del get_versions
