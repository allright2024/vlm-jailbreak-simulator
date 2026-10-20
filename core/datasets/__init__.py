"""Dataset loaders. Importing this package registers all datasets.

Add a new loader module here and import it below so that
``core.get_dataset("<Name>")`` resolves after ``import core.datasets``.
"""

from . import advbench  # noqa: F401  (registers "AdvBench")
from . import advbenchm  # noqa: F401  (registers "AdvBenchM")
from . import cs_dj  # noqa: F401  (registers "CS-DJ")
from . import hades  # noqa: F401  (registers "HADES")
from . import harmbench  # noqa: F401  (registers "HarmBench")
from . import mm_safetybench  # noqa: F401  (registers "MM-SafetyBench")
from . import safebench  # noqa: F401  (registers "SafeBench")

__all__ = [
    "advbench", "advbenchm", "cs_dj", "hades", "harmbench",
    "mm_safetybench", "safebench",
]
