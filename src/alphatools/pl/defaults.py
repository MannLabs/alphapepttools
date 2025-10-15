import matplotlib.colors as mpl_colors


class PlotSettings:
    """Default settings for AlphaTools plots."""

    def __init__(self):
        self.font_family = "sans-serif"
        self.default_font = "Arial"

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
        """Convert PlotSettings to a dictionary."""
        return {key: value for key, value in self.__dict__.items() if not key.startswith("__")}


# Create a global instance
plot_settings = PlotSettings()
