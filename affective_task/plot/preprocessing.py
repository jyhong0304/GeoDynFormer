"""Preprocessing utilities for manuscript figure generation."""

import os
import pickle
import warnings

import numpy as np
from scipy.stats import PearsonRConstantInputWarning

from affective_task import Experiment
from affective_task.model_analysis import (
    FixedPointFinder,
    LatentsLDA,
    LatentSeparation,
)


class Preprocess:
    """Generate model-derived data required by the plotting pipeline.

    For each participant, preprocessing:

    1. Generates model outputs on the test set across noise levels.
    2. Computes model and behavioral summary metrics.
    3. Finds stable fixed points.
    4. Runs the latent-space LDA analysis.
    5. Removes intermediate model-output files that are no longer needed.
    """

    analysis_dir = "model_analysis"
    device = "cpu"
    raw_fn = "data_pre_split.pkl"
    params_fn = "model_params.pth"

    # Noise conditions
    all_noise_sds = np.arange(0.1, 0.65, 0.05)
    all_noise_sds = np.append(all_noise_sds, [0.7, 0.8, 0.9, 1.0])
    all_noise_keys = [
        "01",
        "015",
        "02",
        "025",
        "03",
        "035",
        "04",
        "045",
        "05",
        "055",
        "06",
        "07",
        "08",
        "09",
        "1",
    ]
    latents_noise_keys = [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "1",
    ]
    primary_noise_key = "01"
    primary_noise_sd = 0.1

    # Model and behavioral summary
    outputs_save_str = "holdout_outputs"
    summary_fn = "summary.pkl"
    metrics = [
        "accuracy",
        "acc_switch_cost",
        "mean_rt_trial_type_0",  # G2G
        "mean_rt_trial_type_1",  # E2E
        "mean_rt_trial_type_2",  # E2G
        "mean_rt_trial_type_3",  # G2E
        "switch_cost",
        "switch_cost_trial_type_2",  # E2G switch cost
        "switch_cost_trial_type_3",  # G2E switch cost
        "normed_centroid_dist",
    ]

    # Fixed-point parameters
    fp_fn = "fixed_points.pkl"
    fp_summary_fn = "fixed_point_summary.pkl"
    fp_N = 10
    fp_T = 50000

    # LDA parameters
    lda_fn = "lda_summary.pkl"
    lda_time_range = [-100, 2400]
    lda_n_shuffle = 100

    def __init__(self, model_dir, patients_id, rand_seed, batch_size=None):
        self.model_dir = model_dir
        self.patients_id = patients_id
        self.rand_seed = rand_seed
        self.batch_size = batch_size

        # This warning can occur in calculations that are not included in the
        # summary analyses used by the plotting pipeline.
        warnings.simplefilter("ignore", PearsonRConstantInputWarning)

    def run_preprocessing(self):
        """Run all preprocessing steps for each participant."""
        for patient_id in self.patients_id:
            print("Preprocessing experiment for patient {}".format(patient_id))

            expt_str = "s{}".format(patient_id)
            this_model_dir = os.path.join(self.model_dir, expt_str)

            # Generate model outputs across noise levels.
            self._get_outputs_wrapper(this_model_dir, expt_str)

            # Compute model and behavioral summary metrics, including the
            # normalized distance between task centroids.
            self._get_summary(this_model_dir)

            # Find stable fixed points.
            self._fp_wrapper(this_model_dir, expt_str)

            # Run the latent-space LDA analysis.
            self._lda_wrapper(this_model_dir)

            # Remove intermediate outputs that are no longer required.
            self._clean_up(this_model_dir)

    def _get_outputs_wrapper(self, model_dir, expt_str):
        noise_keys = self.all_noise_keys
        noise_sds = self.all_noise_sds

        for noise_key, noise_sd in zip(noise_keys, noise_sds):
            if noise_key in self.latents_noise_keys:
                analyze_latents = True
            else:
                analyze_latents = False

            self._get_model_outputs(
                model_dir,
                expt_str,
                noise_key,
                noise_sd,
                analyze_latents,
            )

    def _get_model_outputs(
            self,
            model_dir,
            expt_str,
            noise_key,
            noise_sd,
            analyze_latents,
    ):
        save_str = "{}_{}SD.pkl".format(
            self.outputs_save_str,
            noise_key,
        )
        noise_params = {
            "noise_type": "indep",
            "noise_sd": noise_sd,
        }
        expt_kwargs = {
            "logger_type": None,
            "test": noise_params,
            "mode": "testing",
            "params_to_load": self.params_fn,
        }

        # Use the configuration associated with each participant's model.
        config_path = os.path.join(model_dir, "model_config.yaml")

        expt = Experiment(
            model_dir,
            model_dir,
            self.raw_fn,
            expt_str,
            config=config_path,
            processed_dir=model_dir,
            device=self.device,
            **expt_kwargs
        )

        expt.get_behavior_metrics(
            expt.test_dataset,
            save_fn=save_str,
            save_local=True,
            analyze_latents=analyze_latents,
            stats_dir=self.analysis_dir,
            batch_size=self.batch_size,
        )

    def _get_summary(self, model_dir):
        noise_keys = self.all_noise_keys
        noise_sds = self.all_noise_sds

        summary = {key: {} for key in noise_keys}

        for noise_key, noise_sd in zip(noise_keys, noise_sds):
            outputs = self._reload_outputs(model_dir, noise_key)

            if noise_key in self.latents_noise_keys:
                latent_sep = LatentSeparation(outputs)
                dist_stats = latent_sep.analyze()

            for metric in self.metrics:
                if metric == "normed_centroid_dist":
                    summary[noise_key][metric] = dist_stats[metric]
                else:
                    user_key = "u_{}".format(metric)
                    model_key = "m_{}".format(metric)
                    summary[noise_key][user_key] = outputs.summary_stats[user_key]
                    summary[noise_key][model_key] = outputs.summary_stats[model_key]

        save_path = os.path.join(
            model_dir,
            self.analysis_dir,
            self.summary_fn,
        )
        with open(save_path, "wb") as path:
            pickle.dump(summary, path, protocol=4)

    def _fp_wrapper(self, model_dir, expt_str):
        fp_path = os.path.join(
            model_dir,
            self.analysis_dir,
            self.fp_fn,
        )
        fp_summary_path = os.path.join(
            model_dir,
            self.analysis_dir,
            self.fp_summary_fn,
        )

        noise_params = {
            "noise_type": "indep",
            "noise_sd": self.primary_noise_sd,
        }
        expt_kwargs = {
            "logger_type": None,
            "test": noise_params,
            "mode": "testing",
            "params_to_load": self.params_fn,
        }

        expt = Experiment(
            model_dir,
            model_dir,
            self.raw_fn,
            expt_str,
            processed_dir=model_dir,
            device=self.device,
            **expt_kwargs
        )

        outputs = self._reload_outputs(
            model_dir,
            self.primary_noise_key,
        )

        fixed_point_finder = FixedPointFinder(
            expt,
            outputs,
            fp_path,
            fp_summary_path,
            load_saved=False,
            rand_seed=self.rand_seed,
        )
        fixed_points = fixed_point_finder.find_fixed_points(
            self.fp_N,
            self.fp_T,
        )
        fixed_point_finder.get_fixed_point_summary(fixed_points)

    def _lda_wrapper(self, model_dir):
        lda_path = os.path.join(
            model_dir,
            self.analysis_dir,
            self.lda_fn,
        )
        outputs = self._reload_outputs(
            model_dir,
            self.primary_noise_key,
        )

        lda = LatentsLDA(
            outputs,
            lda_path,
            load_saved=False,
            time_range=self.lda_time_range,
            n_shuffle=self.lda_n_shuffle,
            rand_seed=self.rand_seed,
        )
        lda.run_lda_analysis()

    def _reload_outputs(self, model_dir, noise_key):
        outputs_fn = "{}_{}SD.pkl".format(
            self.outputs_save_str,
            noise_key,
        )
        outputs_path = os.path.join(
            model_dir,
            self.analysis_dir,
            outputs_fn,
        )

        with open(outputs_path, "rb") as path:
            outputs = pickle.load(path)

        return outputs

    def _clean_up(self, model_dir):
        for noise_key in self.all_noise_keys:
            if noise_key != self.primary_noise_key:
                outputs_fn = "{}_{}SD.pkl".format(
                    self.outputs_save_str,
                    noise_key,
                )
                outputs_path = os.path.join(
                    model_dir,
                    self.analysis_dir,
                    outputs_fn,
                )
                os.remove(outputs_path)
