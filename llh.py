import numpy as np
from scipy.stats import norm


def compute_llh(
    observed_data,
    model_data,
    im_cols,
    sigma
):
    """
    Compute LLH values for all models.

    Parameters
    ----------
    observed_data : dict  (Observed IM values)

    model_data : dict  (Predicted IM values for each model)

    im_cols : list  (Intensity measure names)

    sigma : array-like  (Standard deviations)

    Returns
    -------
    dict (LLH values)
    """

    llh = {}

    for model in model_data:
        llh[model] = {}

        for i, im in enumerate(im_cols):

            residual = (
                observed_data[im]
                - model_data[model][im]
            )

            llh[model][im] = -np.mean(
                np.log2(
                    norm.pdf(
                        residual,
                        loc=0,
                        scale=sigma[i]
                    )
                )
            )

    return llh


