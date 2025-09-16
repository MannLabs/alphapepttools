class PlotSettings:
    """Default settings for AlphaTools plots."""

    def __init__(self):
        self.font_family = "sans-serif"
        self.default_font = "Arial"

        self.font_sizes = {
            "small": 6,
            "medium": 8,
            "large": 10,
        }

        self.axes = {
            "title_size": 8,
            "label_size": 8,
            "tick_size": 8,
        }

        self.legend = {
            "font_size": 8,
            "title_size": 8,
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
        self.na_color = "#d3d3d3"

    def to_dict(self) -> dict:
        """Convert PlotSettings to a dictionary."""
        return {key: value for key, value in self.__dict__.items() if not key.startswith("__")}


# Create a global instance
plot_settings = PlotSettings()
