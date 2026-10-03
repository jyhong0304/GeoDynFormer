"""Generate Extended Data Figure 8b: non-decision time versus trajectory length."""

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


class FigureS8B:
    """Relate HDDM non-decision time to full model trajectory length."""

    analysis_dir = "model_analysis"
    stats_fn = "holdout_outputs_01SD.pkl"

    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "dataset"
    hddm_stats_fn = "HDDM_group_results_vat_all-trials.csv"

    figsize = (6, 6)
    figdpi = 300

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
            spec["label"]: {
                "trajectory_length": [],
                "non_decision_time": [],
            }
            for spec in self.PANEL_SPECS
        }

    def make_figure(self):
        """Generate, save, and return Extended Data Figure 8b."""
        print("Making Figure S8b...")
        self._run_preprocessing()

        print("Stats for Figure S8b")
        print("--------------------")
        fig = self._plot_figure_get_stats()

        save_figure(fig, self.save_dir, "FigS8b")
        return fig

    def _run_preprocessing(self):
        """Load HDDM NDT values and compute full trajectory lengths."""
        hddm_path = self.dataset_dir / self.hddm_stats_fn
        if not hddm_path.is_file():
            raise FileNotFoundError(
                "Missing required file: {}".format(hddm_path)
            )

        hddm_stats = pd.read_csv(
            hddm_path,
            index_col=0,
        )

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

            patient_ndt = hddm_stats.loc[
                (hddm_stats["subj_id"] == str(patient_id))
                & (hddm_stats["param"] == "non decision time")
                ]

            for spec in self.PANEL_SPECS:
                non_decision_time = patient_ndt.loc[
                    patient_ndt["trial"] == spec["hddm_trial"],
                    "mean",
                ].iloc[0]

                trajectory_length = self._mean_full_trajectory_length(
                    latent_sep,
                    current_task=spec["task"],
                    is_switch=spec["is_switch"],
                )

                data = self.panel_data[spec["label"]]
                data["non_decision_time"].append(non_decision_time)
                data["trajectory_length"].append(trajectory_length)

    @staticmethod
    def _mean_full_trajectory_length(
            latent_sep,
            current_task,
            is_switch,
    ):
        """Return the mean full trajectory length for one task/transition."""
        if is_switch:
            trajectories = latent_sep.get_org_length_full_switch(
                current_task=current_task
            )
        else:
            trajectories = latent_sep.get_org_length_full_repeat(
                current_task=current_task
            )

        return np.mean(np.concatenate(trajectories))

    def _plot_figure_get_stats(self):
        """Assemble the four NDT versus trajectory-length panels."""
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
            self._make_ndt_panel(
                ax,
                trajectory_lengths=data["trajectory_length"],
                non_decision_times=data["non_decision_time"],
                trial_label=spec["label"],
                color=spec["color"],
            )

        return fig

    def _make_ndt_panel(
            self,
            ax,
            trajectory_lengths,
            non_decision_times,
            trial_label,
            color,
            line_ext=0.03,
            plot_boot_band=True,
            n_boot_band=10000,
            ci=95,
            seed=0,
    ):
        """Plot non-decision time against full trajectory length."""
        trajectory_lengths = np.asarray(
            trajectory_lengths,
            dtype=float,
        )
        non_decision_times = np.asarray(
            non_decision_times,
            dtype=float,
        )

        plot_x = np.array(
            [
                non_decision_times.min() - line_ext,
                non_decision_times.max() + line_ext,
            ]
        )

        slope, intercept = np.polyfit(
            non_decision_times,
            trajectory_lengths,
            1,
        )

        ax.plot(
            plot_x,
            slope * plot_x + intercept,
            "k-",
            zorder=2,
            linewidth=0.5,
        )
        ax.scatter(
            non_decision_times,
            trajectory_lengths,
            s=4,
            marker="o",
            zorder=1,
            color=color,
        )

        ax.set_xlabel(
            "Non-decision Time for {}".format(
                trial_label
            )
        )
        ax.set_ylabel(
            "Trajectory Length [0, RT] for {}".format(
                trial_label
            )
        )

        print(
            "Panel B stats: Non-decision Time vs. "
            "Trajectory Length for {}".format(
                trial_label
            )
        )

        r, p_value, ci_low, ci_high = pearson_bootstrap(
            non_decision_times,
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
            "Best-fit slope: {}; intercept: {}".format(
                slope,
                intercept,
            )
        )

        mean_length = np.mean(trajectory_lengths)
        sem_length = (
                np.std(trajectory_lengths)
                / np.sqrt(len(trajectory_lengths))
        )
        print(
            "Mean +/- s.e.m. trajectory length: {} +/- {}".format(
                mean_length,
                sem_length,
            )
        )

        mean_ndt = np.mean(non_decision_times)
        sem_ndt = (
                np.std(non_decision_times)
                / np.sqrt(len(non_decision_times))
        )
        print(
            "Mean +/- s.e.m. Non-decision Time: {} +/- {}".format(
                mean_ndt,
                sem_ndt,
            )
        )
        print("-----------------------------")

        if plot_boot_band:
            y_hat, y_low, y_high = bootstrap_regression_band(
                non_decision_times,
                trajectory_lengths,
                plot_x,
                n_boot=n_boot_band,
                ci=ci,
                seed=seed,
            )

            ax.plot(
                plot_x,
                y_hat,
                linewidth=1,
                color="r",
            )
            ax.fill_between(
                plot_x,
                y_low,
                y_high,
                alpha=0.2,
                color="grey",
            )
