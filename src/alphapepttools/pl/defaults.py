import matplotlib.colors as mpl_colors


class PlotSettings:
    """Central configuration class for alphapepttools plot aesthetics and styling

    Provides a centralized location for managing all plotting defaults used throughout
    alphapepttools. This includes font sizes, marker sizes, line widths, colors, and
    other visual parameters. Settings can be accessed directly or exported as a dictionary
    for serialization and custom configuration.

    The class initializes with sensible defaults optimized for scientific publication
    figures, but all settings can be modified at runtime to customize plot appearance.

    Attributes
    ----------
    font_sizes
        Font size presets with keys 'small' (8pt), 'medium' (10pt), 'large' (12pt)
    axes
        Axes-specific settings for title_size, label_size, and tick_size (all 10pt)
    legend
        Legend settings for font_size and title_size (both 10pt)
    marker_sizes
        Marker size presets with keys 'small' (5), 'medium' (10), 'large' (15)
    linewidths
        Line width presets with keys 'small' (0.25), 'medium' (0.5), 'large' (1.25)
    highlight_colors
        Color presets for highlighting with keys 'high' (#9ecae1), 'low' (#fdae6b),
        'general' (#5ec962)
    resolution
        Output resolution settings with 'dpi' (300)
    preset_sizes
        Figure size presets for different column widths in mm (2-column: 183mm,
        1.5-column: 135mm, 1-column: 89mm, 0.5-column: 45mm, 0.25-column: 22.5mm)
    na_identifiers
        List of strings that identify NA/missing values (default: ['nan'])
    na_color
        RGBA color tuple for representing NA values (default: lightgrey)

    Methods
    -------
    to_dict()
        Convert the PlotSettings instance to a dictionary representation

    Examples
    --------
    Access plot settings globally:

    .. code-block:: python

        from alphapepttools.pl import defaults

        # Access current settings
        print(defaults.plot_settings.font_sizes["large"])  # 12

    Export settings for inspection or serialization:

    .. code-block:: python

        from alphapepttools.pl import defaults

        config = defaults.plot_settings.to_dict()
        print(config.keys())  # Display all available setting categories

    Create a custom PlotSettings instance:

    .. code-block:: python

        from alphapepttools.pl.defaults import PlotSettings

        custom_settings = PlotSettings()
        custom_settings.resolution["dpi"] = 600  # Higher resolution
        custom_settings.linewidths["medium"] = 1.0  # Thicker lines

    Notes
    -----
    A global instance `plot_settings` is created upon module import and is used
    as the default configuration throughout alphapepttools. Modifying this global
    instance will affect all subsequent plots unless explicitly overridden.

    """

    def __init__(self):
        self.font_sizes = {
            "small": 8,
            "medium": 10,
            "large": 12,
        }

        self.axes = {
            "title_size": 10,
            "label_size": 10,
            "tick_size": 10,
        }

        self.legend = {
            "font_size": 10,
            "title_size": 10,
        }

        self.marker_sizes = {
            "small": 5,
            "medium": 10,
            "large": 15,
        }

        self.linewidths = {
            "small": 0.25,
            "medium": 0.5,
            "large": 1.25,
        }

        self.highlight_colors = {
            "high": "#9ecae1",
            "low": "#fdae6b",
            "general": "#5ec962",
        }

        self.resolution = {"dpi": 300}

        self.preset_sizes = {
            "2": 183,
            "1.5": 135,
            "1": 89,
            "0.5": 45,
            "0.25": 22.5,
        }

        # Lookup for NA values for automatic coloring
        self.na_identifiers = ["nan"]
        self.na_color = mpl_colors.to_rgba("lightgrey")

    def to_dict(self) -> dict:
        """Convert PlotSettings instance to a dictionary representation

        Exports all plot settings as a dictionary, excluding any private attributes
        (those starting with double underscores). This is useful for inspecting
        current settings, serializing configuration, or passing settings to other
        plotting functions.

        Returns
        -------
        Dictionary containing all public plot settings with their current values,
        including font_sizes, axes, legend, marker_sizes, linewidths,
        highlight_colors, resolution, preset_sizes, na_identifiers, and na_color.

        Examples
        --------
        Access and inspect current plot settings:

        .. code-block:: python

            from alphapepttools.pl import defaults

            config = defaults.plot_settings.to_dict()
            print(config["font_sizes"])  # {'small': 8, 'medium': 10, 'large': 12}

        Use settings for custom configuration:

        .. code-block:: python

            from alphapepttools.pl import defaults

            # Get current settings
            config = defaults.plot_settings.to_dict()

            # Modify specific settings
            config["font_sizes"]["large"] = 14
            config["highlight_colors"]["high"] = "#ff0000"

        """
        return {key: value for key, value in self.__dict__.items() if not key.startswith("__")}


# Create a global instance
plot_settings = PlotSettings()
