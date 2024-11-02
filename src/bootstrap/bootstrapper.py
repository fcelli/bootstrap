import functools
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from bootstrap import plotting
from bootstrap.utils import check_consistent_lengths


class Bootstrapper:
    def __init__(
        self,
        metric: Callable[..., float],
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64] | None = None,
        n_replicates: int = 10_000,
        random_state: int | None = None,
    ):
        """Initialises the Bootstrapper class.

        Args:
            metric (Callable[..., float]): The function used to compute the
                metric on each resample.
            y_true (NDArray[np.float64]): The array of true data values.
            y_pred (NDArray[np.float64] | None, optional): The array of
                predicted data values. Can be None if computing metrics
                directly on y_true. Defaults to None.
            n_replicates (int, optional): The number of bootstrap replicates to
                generate. Defaults to 10_000.
            random_state (int | None, optional): Seed for random number
                generator, for reproducibility. Defaults to None.
        """
        check_consistent_lengths(y_true, y_pred)

        if n_replicates <= 0:
            raise ValueError(
                "Number of replicates must be a positive integer."
            )
        if not callable(metric):
            raise ValueError("The metric function must be callable.")

        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred) if y_pred is not None else y_pred
        self.metric = metric
        self.n_replicates = n_replicates
        self._bootstrap_replicates: NDArray[np.float64] | None = None

        if random_state is not None:
            np.random.seed(random_state)

    @functools.cached_property
    def bootstrap_replicates(self) -> NDArray[np.float64]:
        """Gets the bootstrap replicates, computing them if they have not been
        generated yet.

        Returns:
            NDArray[np.float64]: Array of bootstrapped metric values.
        """
        sample_size = len(self.y_true)
        indexes = np.random.randint(
            0, sample_size, self.n_replicates * sample_size
        )
        r_true = self.y_true[indexes].reshape(self.n_replicates, sample_size)
        if self.y_pred is None:
            bootstrap_replicates = np.apply_along_axis(self.metric, 1, r_true)
        else:
            r_pred = self.y_pred[indexes].reshape(
                self.n_replicates, sample_size
            )
            stacked = np.stack((r_true, r_pred), axis=1)
            bootstrap_replicates = np.array(
                [self.metric(s[0], s[1]) for s in stacked]
            )
        return bootstrap_replicates

    @property
    def sample_metric(self) -> float:
        """Computes the metric on the original data sample.

        Returns:
            float: The metric value for the original data sample.
        """
        return (
            self.metric(self.y_true)
            if self.y_pred is None
            else self.metric(self.y_true, self.y_pred)
        )

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
        return tuple(
            np.percentile(
                self.bootstrap_replicates, [lower_bound, upper_bound]
            )
        )

    def plot(self) -> None:
        plotting.sampling_distribution_histogram(
            self.bootstrap_replicates,
            self.sample_metric,
        )
