from typing import List, Sequence

import numpy as np
from scipy.stats import uniform


def uniform_ND(limits: Sequence[Sequence[int]], n_draw: int) -> List[List[int]]:
    """
    limits = [[limit_x1, limit_y1],
              [limit_x2, limit_y2],
              ...]
    returns an array with uniformly sampled random variables
    return array is of shape (n_requested_samples, n_dimensions)
    """
    dim = np.array(limits).shape[0]

    rvars = np.empty([n_draw, dim])

    for d, limit in enumerate(limits):
        lower = np.min(limit)
        upper = np.max(limit)
        # print(lower)
        # print(upper)
        # "distribution is constant between loc and loc + scale.""
        rv = uniform(lower, upper - lower)

        # rvar = rvs(n_draw)

        rvars[:, d] = rv.rvs(n_draw)

    return rvars
