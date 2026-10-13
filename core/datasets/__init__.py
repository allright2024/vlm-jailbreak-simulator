"""Dataset loaders. Importing this package registers all datasets.

Add a new loader module here and import it below so that
``core.get_dataset("<Name>")`` resolves after ``import core.datasets``.
"""

from . import advbench  # noqa: F401  (registers "AdvBench")
from . import advbenchm  # noqa: F401  (registers "AdvBenchM")
from . import hades  # noqa: F401  (registers "HADES")
from . import mm_safetybench  # noqa: F401  (registers "MM-SafetyBench")
from . import safebench  # noqa: F401  (registers "SafeBench")

__all__ = ["advbench", "advbenchm", "hades", "mm_safetybench", "safebench"]
