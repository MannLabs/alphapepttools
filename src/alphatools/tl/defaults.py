from typing import ClassVar


class TLDefaults:
    """Default values and constants for the tl module."""

    DIFF_EXP_COLS: ClassVar[list[str]] = [
        "condition_pair",
        "protein",
        "log2fc",
        "p_value",
        "-log10(p_value)",
        "fdr",
        "-log10(fdr)",
        "method",
        "max_level_1_samples",
        "max_level_2_samples",
    ]

    CEILING_NEGATIVE_LOG10_PVALUE: ClassVar[float] = 300.0


# Create a singleton instance for easy import
tl_defaults = TLDefaults()
