from enum import Enum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


class Colours(str, Enum):
    black = "#000000"
    blue = "#0059ab"
    red = "#bf0000"


def sampling_distribution_histogram(
    bootstrap_replicates: NDArray[np.float64],
    sample_metric: float | None = None,
    hist_kwargs: dict[str, Any] | None = None,
    sm_vline_kwargs: dict[str, Any] | None = None,
) -> None:
    """Plots the sampling distribution."""
    hist_settings: dict[str, Any] = {
        "bins": 10,
        "alpha": 1,
        "color": Colours.blue,
        "edgecolor": Colours.black,
        "label": "Sampling Distribution",
    }
    sm_vline_settings: dict[str, Any] = {
        "color": Colours.red,
        "linestyle": "dashed",
        "linewidth": 2,
        "label": "Sample Metric",
    }

    if hist_kwargs is not None:
        hist_settings.update(hist_kwargs)
    if sm_vline_kwargs is not None:
        sm_vline_settings.update(sm_vline_kwargs)

    plt.hist(bootstrap_replicates, **hist_settings)
    if sample_metric is not None:
        plt.axvline(sample_metric, **sm_vline_settings)
    plt.title("Bootstrap Sampling Distribution")
    plt.xlabel("Metric")
    plt.ylabel("Count")
    plt.legend()
    plt.show()
