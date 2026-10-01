"""Generate Extended Data Figure 3: model-versus-participant behavior."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from affective_task.utils import plot_scatter, save_figure


class FigureS3:
    """Compare participant and model RT/switch-cost summaries."""

    analysis_dir = "model_analysis"
    stats_fn = "holdout_outputs_01SD.pkl"

    figsize = (8, 12)
    figdpi = 300
    line_ext = 10

    SCATTER_SPECS = [
        {
            "grid": (slice(0, 6), slice(0, 6)),
            "ax_lims": [900, 2100],
            "metric": "mean_rt_trial_type_0",
            "label": "mean RT for G2G",
        },
        {
            "grid": (slice(0, 6), slice(9, 15)),
            "ax_lims": [950, 2050],
            "metric": "mean_rt_trial_type_1",
            "label": "mean RT for E2E",
        },
        {
            "grid": (slice(8, 14), slice(0, 6)),
            "ax_lims": [1000, 2800],
            "metric": "mean_rt_trial_type_2",
            "label": "mean RT for E2G",
        },
        {
            "grid": (slice(8, 14), slice(9, 15)),
            "ax_lims": [1200, 2900],
            "metric": "mean_rt_trial_type_3",
            "label": "mean RT for G2E",
        },
        {
            "grid": (slice(16, 22), slice(0, 6)),
            "ax_lims": [-500, 1400],
            "metric": "switch_cost_trial_type_2",
            "label": "switch cost for E2G",
        },
        {
            "grid": (slice(16, 22), slice(9, 15)),
            "ax_lims": [-50, 1500],
            "metric": "switch_cost_trial_type_3",
            "label": "switch cost for G2E",
        },
    ]

    GROUP_STAT_KEYS = [
        "m_mean_rt_trial_type_0",
        "u_mean_rt_trial_type_0",
        "m_mean_rt_trial_type_1",
        "u_mean_rt_trial_type_1",
        "m_mean_rt_trial_type_2",
        "u_mean_rt_trial_type_2",
        "m_mean_rt_trial_type_3",
        "u_mean_rt_trial_type_3",
        "m_switch_cost_trial_type_2",
        "u_switch_cost_trial_type_2",
        "m_switch_cost_trial_type_3",
        "u_switch_cost_trial_type_3",
    ]

    def __init__(
            self,
            model_dir,
            save_dir,
            patients_id,
            rand_seed,
            n_boot,
    ):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.patients_id = patients_id
        self.rng = np.random.default_rng(
            rand_seed
        )
        self.n_boot = n_boot
        self.alpha = 0.05

        self.group_stats = {
            key: []
            for key in self.GROUP_STAT_KEYS
        }

    def make_figure(self):
        """Generate, save, and return Extended Data Figure 3."""
        print("Making Figure S3...")
        self._run_preprocessing()

        print("Stats for Figure S3")
        print("-------------------")

        fig = self._plot_figure_get_stats()

        save_figure(
            fig,
            self.save_dir,
            "FigS3",
        )
        return fig

    def _run_preprocessing(self):
        """Collect participant/model summary statistics for all subjects."""
        for patient_id in self.patients_id:
            stats_path = (
                    Path(self.model_dir)
                    / "s{}".format(patient_id)
                    / self.analysis_dir
                    / self.stats_fn
            )

            with stats_path.open("rb") as file:
                expt_stats = pickle.load(file)

            for key in self.GROUP_STAT_KEYS:
                self.group_stats[key].append(
                    expt_stats.summary_stats[key]
                )

    def _plot_figure_get_stats(self):
        """Assemble the six participant-versus-model scatter plots."""
        fig = plt.figure(
            constrained_layout=False,
            figsize=self.figsize,
            dpi=self.figdpi,
        )
        gs = fig.add_gridspec(
            22,
            15,
        )

        for spec in self.SCATTER_SPECS:
            row_slice, col_slice = spec[
                "grid"
            ]
            ax = fig.add_subplot(
                gs[
                    row_slice,
                    col_slice,
                ]
            )

            params = {
                "ax_lims": spec["ax_lims"],
                "metric": spec["metric"],
                "label": spec["label"],
            }

            plot_scatter(
                self.group_stats,
                params,
                ax,
                self.line_ext,
                self.rng,
                n_boot=self.n_boot,
                alpha=self.alpha,
            )

        return fig
