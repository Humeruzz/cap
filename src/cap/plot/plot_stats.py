import warnings

warnings.warn(
    "cap.plot.plot_stats is deprecated and will be removed in a future version. "
    "Use cap.plot.stats_html instead.",
    DeprecationWarning,
    stacklevel=2,
)

from cap.plot.stats_html import *  # noqa: F401, F403, E402
from cap.plot.stats_html import create_interactive_viewer  # noqa: F401, E402
