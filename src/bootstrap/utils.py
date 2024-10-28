import numpy as np
from numpy.typing import NDArray


def check_consistent_lengths(*arrays: NDArray[np.float64] | None) -> None:
    """Checks that all arrays are of the same length or None."""
    lengths = [len(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(length) for length in lengths]
        )
