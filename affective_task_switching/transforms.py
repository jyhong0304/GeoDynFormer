"""Data transformations used by the affective task-switching dataset.

Each transformation is implemented as a callable class and can be supplied to
``AffectiveTaskDataset`` during preprocessing or online training.
"""

import math

import numpy as np
import torch

from .utils import median_absolute_dev as mabsdev


class _Trim:
    """Trim continuous trial data to the configured analysis window.

    This transform is applied by default during dataset processing.
    """

    def __call__(self, data):
        for key in data.continuous.keys():
            data.continuous[key] = data.continuous[key][
                                   : data.num_steps_short_win, :
                                   ]
        return data


class SmoothResponses:
    """Smooth the continuous representation of the user's responses.

    Responses are smoothed with a Gaussian kernel whose standard deviation is
    specified by ``params["kernel_sd"]``. The same kernel is used for all four
    trial types: G2G, E2E, E2G, and G2E.
    """

    kernel_t_max = 3520
    NUM_OF_TRIAL_TYPES = 4

    def _build_sm_kernel(self, params):
        """Build the Gaussian response-smoothing kernel."""
        t = np.arange(0, self.kernel_t_max, params["step_size"])
        sd = params["kernel_sd"]
        mean = self.kernel_t_max / 2

        single_kernel = np.exp(-0.5 * ((t - mean) / sd) ** 2)
        single_kernel = single_kernel / np.amax(single_kernel)

        self.kernel = [
            single_kernel for _ in range(self.NUM_OF_TRIAL_TYPES)
        ]

    def __call__(self, data):
        """Smooth the response signals of a single trial sequence.

        Args
        ----
        data (AffectiveTaskTrialData instance): Trial data to transform.

        Returns
        -------
        data (AffectiveTaskTrialData instance): Trial data after smoothing.
        """
        responses = data.continuous["uresp_ans"]
        sm_responses = np.zeros(responses.shape[:2])

        for task_ans in range(responses.shape[1]):
            # Response channels: male, female, happy, threatening.
            for trial_type in range(responses.shape[2]):
                # Trial types:
                # 0 = G2G
                # 1 = E2E
                # 2 = E2G
                # 3 = G2E
                conv_full = np.convolve(
                    responses[:, task_ans, trial_type],
                    self.kernel[trial_type],
                    mode="same",
                )
                sm_responses[:-1, task_ans] += conv_full[1:]

        data.continuous["uresp_ans"] = sm_responses
        return data


class FilterOutliers:
    """Mark trial sequences containing outlier response times as invalid.

    Outlier handling is configured through ``params["outlier_params"]``.

    Parameters
    ----------
    method : str
        Outlier-detection method. The currently supported method is ``"mad"``,
        based on the median absolute deviation (MAD).
    thresh : float
        Number of MAD units used as the outlier threshold.
    mad : float
        Dataset-level median absolute deviation. Required for ``"mad"``.
    median : float
        Dataset-level median response time. Required for ``"mad"``.

    Notes
    -----
    If any response time in an ``AffectiveTaskTrialData`` instance is flagged
    as an outlier, the entire instance is marked invalid.
    """

    supported_methods = ["mad"]

    def __init__(self, params):
        self.method = params["outlier_params"]["method"]
        assert (
                self.method in self.supported_methods
        ), "Outlier method not supported!"
        self.thresh = params["outlier_params"]["thresh"]

        if self.method == "mad":
            self.mad = params["outlier_params"]["mad"]
            self.median = params["outlier_params"]["median"]

    def __call__(self, data):
        """Apply response-time outlier filtering to a single trial sequence.

        Args
        ----
        data (AffectiveTaskTrialData instance): Trial data to transform.

        Returns
        -------
        data (AffectiveTaskTrialData instance): Trial data after filtering.
            If an outlier is detected, ``data.is_valid`` is set to ``False``.
        """
        rts = data.discrete["urt_ms"]

        if len(rts) == 0:
            data.is_valid = False
        else:
            if self.method == "mad":
                _, devs = mabsdev(np.array(rts), median=self.median)
                if len(devs[devs >= self.thresh * self.mad]) > 0:
                    data.is_valid = False  # Entire trial sequence is excluded.

        return data


class AddStimulusNoise:
    """Add Gaussian noise to the continuous stimulus representation.

    Noise behavior is controlled by the following parameters:

    ``noise_type``
        ``"corr"`` for correlated noise or ``"indep"`` for independent noise.
    ``noise_sd``
        Standard deviation of the Gaussian noise.
    ``noise_corr_weight``
        Strength of the cross-channel correlation when ``noise_type="corr"``.
        Values should lie between 0 and 1.
    """

    def __init__(self, params):
        self.params = params

    def __call__(self, data):
        """Add stimulus noise to a single model-input sequence.

        Args
        ----
        data (torch.Tensor): Continuous model inputs.

        Returns
        -------
        noisy_data (torch.Tensor): Copy of the inputs after noise is added.
        """
        noisy_data = torch.clone(data)

        if self.params["noise_type"] == "corr":
            noisy_data = self._add_correlated_noise(noisy_data)
        elif self.params["noise_type"] == "indep":
            noisy_data = self._add_indep_noise(noisy_data)

        return noisy_data

    def _add_indep_noise(self, data):
        std = self.params["noise_sd"] * torch.ones(data[:, 4:].shape)
        noise = torch.normal(mean=0.0, std=std)
        data[:, 4:] = (data[:, 4:] + noise).to(dtype=torch.float32)
        return data

    def _add_correlated_noise(self, data, latent_weights=[0.6, 0.2, 0.2]):
        noise_cw = self.params["noise_corr_weight"]
        n_time = data.shape[0]

        latent_inds = [(0, 1, 2), (1, 0, 2), (2, 0, 1)]
        n_dims = [4, 4, 2]
        start_inds = [4, 8, 12]

        latent_noise_sd = self.params["noise_sd"] * torch.ones((n_time, 3))
        latent_sources = torch.normal(mean=0.0, std=latent_noise_sd)

        for li, nd, si in zip(latent_inds, n_dims, start_inds):
            this_corr_noise = (
                    math.sqrt(latent_weights[0]) * latent_sources[:, li[0]]
                    + math.sqrt(latent_weights[1]) * latent_sources[:, li[1]]
                    + math.sqrt(latent_weights[2]) * latent_sources[:, li[2]]
            )
            this_corr_noise = this_corr_noise.unsqueeze(1).repeat(1, nd)

            this_noise_sd = self.params["noise_sd"] * torch.ones((n_time, nd))
            this_indep_noise = torch.normal(mean=0.0, std=this_noise_sd)

            # This scaling keeps the resulting noise standard deviation
            # approximately equal to ``self.params["noise_sd"]``.
            this_noise = (
                    math.sqrt(noise_cw) * this_corr_noise
                    + math.sqrt(1 - noise_cw) * this_indep_noise
            )
            data[:, si: si + nd] = (data[:, si: si + nd] + this_noise).to(
                dtype=torch.float32
            )

        return data


class Compose:
    """Apply a sequence of transformations in order.

    Args
    ----
    transforms (list of callables): Transformations to apply sequentially.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        """Apply all configured transformations to ``data``."""
        for t in self.transforms:
            data = t(data)
        return data
