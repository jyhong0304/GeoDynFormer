"""Generate Extended Data Figure 8a: drift rate versus trajectory length."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from affective_task.model_analysis import LatentSeparation
from affective_task.utils import (
    Constants,
    bootstrap_regression_band,
    pearson_bootstrap,
    p_to_stars,
    save_figure,
)


class FigureS8A:
    """Relate HDDM drift rate to model trajectory length across trial types."""

    analysis_dir = "model_analysis"
    stats_fn = "holdout_outputs_01SD.pkl"

    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "dataset"
    hddm_stats_fn = "HDDM_group_results_vat_all-trials.csv"

    figsize = (6, 6)
    figdpi = 300
    START_T_MS = 100

    PANEL_SPECS = [
        {
            "grid": (slice(0, 4), slice(0, 4)),
            "label": "G2G",
            "task": 0,
            "is_switch": False,
            "hddm_trial": "Gender Repeat Trials",
            "color": Constants.COLOR_G2G,
        },
        {
            "grid": (slice(0, 4), slice(5, 9)),
            "label": "E2E",
            "task": 1,
            "is_switch": False,
            "hddm_trial": "Emotion Repeat Trials",
            "color": Constants.COLOR_E2E,
        },
        {
            "grid": (slice(5, 9), slice(0, 4)),
            "label": "E2G",
            "task": 0,
            "is_switch": True,
            "hddm_trial": "Gender Switch Trials",
            "color": Constants.COLOR_E2G,
        },
        {
            "grid": (slice(5, 9), slice(5, 9)),
            "label": "G2E",
            "task": 1,
            "is_switch": True,
            "hddm_trial": "Emotion Switch Trials",
            "color": Constants.COLOR_G2E,
        },
    ]

    def __init__(self, model_dir, save_dir, patients_id, rand_seed, n_boot):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.patients_id = patients_id
        self.rng = np.random.default_rng(rand_seed)
        self.n_boot = n_boot
        self.alpha = 0.05

        self.panel_data = {
            spec["label"]: {"trajectory_length": [], "drift_rate": []}
            for spec in self.PANEL_SPECS
        }

    def make_figure(self):
        """Generate, save, and return Extended Data Figure 8a."""
        print("Making Figure S8a...")
        self._run_preprocessing()

        print("Stats for Figure S8a")
        print("--------------------")
        fig = self._plot_figure_get_stats()

        save_figure(fig, self.save_dir, "FigS8a")
        return fig

    def _run_preprocessing(self):
        """Load drift rates and compute trajectory lengths for each subject."""
        hddm_path = self.dataset_dir / self.hddm_stats_fn
        if not hddm_path.is_file():
            raise FileNotFoundError("Missing required file: {}".format(hddm_path))

        hddm_stats = pd.read_csv(hddm_path, index_col=0)

        for patient_id in self.patients_id:
            stats_path = (
                    Path(self.model_dir)
                    / "s{}".format(patient_id)
                    / self.analysis_dir
                    / self.stats_fn
            )

            with stats_path.open("rb") as file:
                expt_stats = pickle.load(file)

            latent_sep = LatentSeparation(expt_stats)
            patient_drift = hddm_stats.loc[
                (hddm_stats["subj_id"] == str(patient_id))
                & (hddm_stats["param"] == "drift rate")
                ]

            for spec in self.PANEL_SPECS:
                drift_rate = patient_drift.loc[
                    patient_drift["trial"] == spec["hddm_trial"],
                    "mean",
                ].iloc[0]

                trajectory_length = self._mean_trajectory_length(
                    latent_sep,
                    current_task=spec["task"],
                    is_switch=spec["is_switch"],
                )

                data = self.panel_data[spec["label"]]
                data["drift_rate"].append(drift_rate)
                data["trajectory_length"].append(trajectory_length)

    def _mean_trajectory_length(self, latent_sep, current_task, is_switch):
        """Return mean trajectory length using the original 100-ms start."""
        if is_switch:
            trajectories = latent_sep.get_org_length_starting_specific_switch(
                current_task=current_task,
                start_t=self.START_T_MS,
            )
        else:
            trajectories = latent_sep.get_org_length_starting_specific_repeat(
                current_task=current_task,
                start_t=self.START_T_MS,
            )

        return np.mean(np.concatenate(trajectories))

    def _plot_figure_get_stats(self):
        """Assemble the four drift-rate versus trajectory-length panels."""
        fig = plt.figure(
            constrained_layout=False,
            figsize=self.figsize,
            dpi=self.figdpi,
        )
        gs = fig.add_gridspec(9, 9)

        for spec in self.PANEL_SPECS:
            rows, cols = spec["grid"]
            ax = fig.add_subplot(gs[rows, cols])
            data = self.panel_data[spec["label"]]

            self._make_drift_panel(
                ax,
                trajectory_lengths=data["trajectory_length"],
                drift_rates=data["drift_rate"],
                trial_label=spec["label"],
                color=spec["color"],
            )

        return fig

    def _make_drift_panel(
            self,
            ax,
            trajectory_lengths,
            drift_rates,
            trial_label,
            color,
            line_ext=0.03,
            plot_boot_band=True,
            n_boot_band=10000,
            ci=95,
            seed=0,
    ):
        """Plot drift rate against trajectory length for one trial type."""
        trajectory_lengths = np.asarray(trajectory_lengths, dtype=float)
        drift_rates = np.asarray(drift_rates, dtype=float)

        plot_x = np.array(
            [drift_rates.min() - line_ext, drift_rates.max() + line_ext]
        )
        slope, intercept = np.polyfit(drift_rates, trajectory_lengths, 1)

        ax.plot(
            plot_x,
            slope * plot_x + intercept,
            "k-",
            zorder=2,
            linewidth=0.5,
        )
        ax.scatter(
            drift_rates,
            trajectory_lengths,
            s=4,
            marker="o",
            zorder=1,
            color=color,
        )
        ax.set_xlabel("Drift rate for {}".format(trial_label))
        ax.set_ylabel(
            "Trajectory Length [0, RT] for {}".format(trial_label)
        )

        print(
            "Panel A stats: Drift rate vs. Trajectory Length "
            "[0, RT] for {}".format(trial_label)
        )

        r, p_value, ci_low, ci_high = pearson_bootstrap(
            drift_rates,
            trajectory_lengths,
            self.rng,
            n_boot=self.n_boot,
            alpha=self.alpha,
        )

        print(
            "r = {}, 95% CI: ({}, {}), p = {:0.2e} ({})".format(
                round(r, 2),
                round(ci_low, 2),
                round(ci_high, 2),
                p_value,
                p_to_stars(p_value),
            )
        )
        print(
            "Best-fit slope: {}; intercept: {}".format(slope, intercept)
        )

        mean_length = np.mean(trajectory_lengths)
        sem_length = np.std(trajectory_lengths) / np.sqrt(
            len(trajectory_lengths)
        )
        print(
            "Mean +/- s.e.m. trajectory length: {} +/- {}".format(
                mean_length,
                sem_length,
            )
        )

        mean_drift = np.mean(drift_rates)
        sem_drift = np.std(drift_rates) / np.sqrt(len(drift_rates))
        print(
            "Mean +/- s.e.m. drift rate: {} +/- {}".format(
                mean_drift,
                sem_drift,
            )
        )
        print("-----------------------------")

        if plot_boot_band:
            y_hat, y_low, y_high = bootstrap_regression_band(
                drift_rates,
                trajectory_lengths,
                plot_x,
                n_boot=n_boot_band,
                ci=ci,
                seed=seed,
            )
            ax.plot(plot_x, y_hat, linewidth=1, color="r")
            ax.fill_between(
                plot_x,
                y_low,
                y_high,
                alpha=0.2,
                color="grey",
            )
