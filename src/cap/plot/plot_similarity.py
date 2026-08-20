import warnings

warnings.warn(
    "cap.plot.plot_similarity is deprecated and will be removed in a future version. "
    "Use cap.plot.similarity_html instead.",
    DeprecationWarning,
    stacklevel=2,
)

from cap.plot.similarity_html import *  # noqa: F403, E402
from cap.plot.similarity_html import create_similarity_viewer  # noqa: F401, E402
