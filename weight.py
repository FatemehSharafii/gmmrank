import numpy as np


def llh_weights(llh):
    """
    Convert LLH values to normalized weights.
    """

    ims = next(iter(llh.values())).keys()

    weights = {}

    for im in ims:

        numerators = {
            model: 2 ** (-llh[model][im])
            for model in llh
        }

        denominator = sum(numerators.values())

        weights[im] = {
            model: value / denominator
            for model, value in numerators.items()
        }

    return weights







