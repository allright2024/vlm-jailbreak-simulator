"""Attack registration hub.

The attack implementations live in per-method packages under ``multi_attacks/``
(and ``image_attacks/visual_adv`` for the pure image attack). Importing this
module imports each of them so their ``@register_attack`` decorators run; the
pipeline keeps doing ``import core.attacks`` to make every attack resolvable via
``core.get_attack(...)``.
"""

import image_attacks.delta_jp    # noqa: F401  (registers "DeltaJP")
import image_attacks.visual_adv  # noqa: F401  (registers "VisualAdv")
import multi_attacks.cs_dj       # noqa: F401  (registers "CS-DJ")
import multi_attacks.figstep     # noqa: F401  (registers "FigStep")
import multi_attacks.hades       # noqa: F401  (registers "HADES")
import multi_attacks.jood        # noqa: F401  (registers "JOOD")
import multi_attacks.jps         # noqa: F401  (registers "JPS")
import multi_attacks.jailbreak_in_pieces  # noqa: F401  (registers "JailbreakInPieces")
import multi_attacks.jailbound   # noqa: F401  (registers "JailBound")
import multi_attacks.mml         # noqa: F401  (registers "MML")
import multi_attacks.ultrabreak  # noqa: F401  (registers "UltraBreak")

__all__: list[str] = []
