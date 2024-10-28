import numpy as np
from typing import Callable
from numpy.typing import NDArray


def check_consistent_lengths(*arrays: NDArray[np.float64] | None) -> None:
    lengths = [len(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(length) for length in lengths]
        )


class Bootstrapper:
    def __init__(
        self,
        metric: Callable[..., float],
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64] | None = None,
        n_replicates: int = 10_000,
    ):
        check_consistent_lengths(y_true, y_pred)

        self._sample_metric = (
            metric(y_true) if y_pred is None else metric(y_true, y_pred)
        )
        sample_size = len(y_true)
        indexes = np.random.randint(0, sample_size, n_replicates * sample_size)
        r_true = y_true[indexes].reshape(n_replicates, sample_size)
        if y_pred is None:
            self._results = np.apply_along_axis(metric, 1, r_true)
        else:
            r_pred = y_pred[indexes].reshape(n_replicates, sample_size)
            stacked = np.stack((r_true, r_pred), axis=1)
            self._results = np.array([metric(r[0], r[1]) for r in stacked])

    @property
    def sample_metric(self) -> float:
        return self._sample_metric

    @property
    def results(self) -> NDArray[np.float64]:
        return self._results

    def confidence_interval(
        self, percentile: float = 95
    ) -> tuple[float, float]:
        """Calculates the confidence interval for the bootstrapped metric.

        Args:
            percentile (float, optional): The desired confidence level.
                Defaults to 95.

        Returns:
            tuple[float, float]: The lower and upper bounds of the confidence
                interval.
        """
        if not (0 < percentile < 100):
            raise ValueError("Percentile must be between 0 and 100.")

        lower_bound = (100 - percentile) / 2
        upper_bound = 100 - lower_bound
        return tuple(np.percentile(self.results, [lower_bound, upper_bound]))
