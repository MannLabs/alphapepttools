from typing import Literal

from scverse_misc import Settings


class AlphaPeptToolsSettings(Settings):
    """Settings for alphapepttools

    Update with `alphapepttools.settings.<key> = <value>`
    """

    plotting_api: Literal["v1", "v2"] = "v1"


settings = AlphaPeptToolsSettings()
