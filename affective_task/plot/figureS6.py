"""Generate Extended Data Figure 6: distance-RT correlation distributions."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_1samp

from affective_task.model_analysis import LatentSeparation
from affective_task.utils import Constants, p_to_stars, save_figure


class FigureS6:
    """Plot across-subject distributions of latent-distance/RT correlations."""

    analysis_dir = "model_analysis"
    stats_fn = "holdout_outputs_01SD.pkl"

    figsize = (6, 4)
    figdpi = 300

    PANEL_SPECS = [
        {
            "name": "A",
            "grid": (slice(0, 2), slice(0, 2)),
            "label": "all repeat trials",
            "color": Constants.COLOR_REPEAT,
            "xlabel": (
                "Mean Pearson's r between\n"
                "trajectory length and RT (all repeat trials)"
            ),
            "yticks": range(0, 13),
            "analysis": ("repeat_all", None),
        },
        {
            "name": "A1",
            "grid": (slice(0, 2), slice(4, 6)),
            "label": "G2G",
            "color": Constants.COLOR_G2G,
            "xlabel": (
                "Mean Pearson's r between\n"
                "trajectory length and RT (G2G)"
            ),
            "yticks": range(0, 9),
            "analysis": ("repeat_task", 0),
        },
        {
            "name": "A2",
            "grid": (slice(0, 2), slice(8, 10)),
            "label": "E2E",
            "color": Constants.COLOR_E2E,
            "xlabel": (
                "Mean Pearson's r between\n"
                "trajectory length and RT (E2E)"
            ),
            "yticks": range(0, 13),
            "analysis": ("repeat_task", 1),
        },
        {
            "name": "B",
            "grid": (slice(4, 6), slice(0, 2)),
            "label": "all switch trials",
            "color": Constants.COLOR_SWITCH,
            "xlabel": (
                "Mean Pearson's r between\n"
                "trajectory length and RT (all switch trials)"
            ),
            "yticks": range(0, 8),
            "analysis": ("switch_all", None),
        },
        {
            "name": "B1",
            "grid": (slice(4, 6), slice(4, 6)),
            "label": "E2G",
            "color": Constants.COLOR_E2G,
            "xlabel": (
                "Mean Pearson's r between\n"
                "trajectory length and RT (E2G)"
            ),
            "yticks": range(0, 6),
            "analysis": ("switch_task", 0),
        },
        {
            "name": "B2",
            "grid": (slice(4, 6), slice(8, 10)),
            "label": "G2E",
            "color": Constants.COLOR_G2E,
            "xlabel": (
                "Mean Pearson's r between\n"
                "trajectory length and RT (G2E)"
            ),
            "yticks": range(0, 6),
            "analysis": ("switch_task", 1),
        },
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

        # Retain the common figure-class constructor signature.
        _ = rand_seed, n_boot

        self.panel_results = {
            spec["name"]: {
                "stats": [],
                "num_low_N": [],
            }
            for spec in self.PANEL_SPECS
        }

    def make_figure(self):
        """Generate, save, and return Extended Data Figure 6."""
        print("Making Figure S6...")
        self._run_preprocessing()

        print("Stats for Figure S6")
        print("-------------------")

        fig = self._plot_figure_get_stats()

        save_figure(
            fig,
            self.save_dir,
            "FigS6",
        )
        return fig

    def _run_preprocessing(self):
        """Compute distance-RT correlation summaries for each subject."""
        for patient_id in self.patients_id:
            stats_path = (
                    Path(self.model_dir)
                    / "s{}".format(patient_id)
                    / self.analysis_dir
                    / self.stats_fn
            )

            with stats_path.open("rb") as file:
                expt_stats = pickle.load(file)

            latent_sep = LatentSeparation(
                expt_stats
            )

            for spec in self.PANEL_SPECS:
                result = self._run_latent_separation_analysis(
                    latent_sep,
                    spec["analysis"],
                )

                self.panel_results[
                    spec["name"]
                ]["stats"].append(
                    result["mean_r"]
                )
                self.panel_results[
                    spec["name"]
                ]["num_low_N"].append(
                    result["num_low_N"]
                )

    @staticmethod
    def _run_latent_separation_analysis(
            latent_sep,
            analysis_spec,
    ):
        """Dispatch one LatentSeparation analysis used by the figure."""
        analysis_name, current_task = analysis_spec

        if analysis_name == "repeat_all":
            return latent_sep.analyze_repeat()

        if analysis_name == "repeat_task":
            return latent_sep.analyze_latent_dist_repeat(
                current_task=current_task
            )

        if analysis_name == "switch_all":
            return latent_sep.analyze_latent_dist()

        if analysis_name == "switch_task":
            return latent_sep.analyze_latent_dist_switch(
                current_task=current_task
            )

        raise KeyError(
            "Unknown analysis type: {}".format(
                analysis_name
            )
        )

    def _plot_figure_get_stats(self):
        """Assemble the six correlation-distribution histograms."""
        fig = plt.figure(
            constrained_layout=False,
            figsize=self.figsize,
            dpi=self.figdpi,
        )
        gs = fig.add_gridspec(
            6,
            10,
        )

        for spec in self.PANEL_SPECS:
            row_slice, col_slice = spec[
                "grid"
            ]
            ax = fig.add_subplot(
                gs[
                    row_slice,
                    col_slice,
                ]
            )

            results = self.panel_results[
                spec["name"]
            ]

            self._make_histogram_panel(
                ax=ax,
                panel_name=spec["name"],
                label=spec["label"],
                stats_values=results["stats"],
                num_low_N=results["num_low_N"],
                color=spec["color"],
                xlabel=spec["xlabel"],
                yticks=spec["yticks"],
            )

        return fig

    def _make_histogram_panel(
            self,
            ax,
            panel_name,
            label,
            stats_values,
            num_low_N,
            color,
            xlabel,
            yticks,
    ):
        """Plot one correlation histogram and print its summary statistics."""
        self._print_exclusion_stats(
            panel_name=panel_name,
            label=label,
            num_low_N=num_low_N,
        )

        values = np.asarray(
            stats_values,
            dtype=float,
        )
        values = values[
            ~np.isnan(values)
        ]

        n = len(values)
        t_stat, p_value = ttest_1samp(
            values,
            popmean=0.0,
        )

        n_positive = np.count_nonzero(
            values > 0
        )
        mean_value = np.mean(
            values
        )
        sem_value = (
                np.std(
                    values,
                    ddof=1,
                )
                / np.sqrt(n)
        )

        print(
            "Panel {}, main summary stats for {}".format(
                panel_name,
                label,
            )
        )
        print(
            "Mean +/- s.e.m. corr. within model, dist vs. model RTs: "
            "{} +/- {}".format(
                mean_value,
                sem_value,
            )
        )
        print(
            "One-sample t-test against zero: p = {}, t({}) = {}".format(
                p_value,
                n - 1,
                t_stat,
            )
        )
        print(
            "Total models analyzed: {}".format(
                n
            )
        )
        print(
            "Num. models with positive corr.: {}".format(
                n_positive
            )
        )
        print(
            "Fraction of models with positive corr.: {}".format(
                n_positive / n
            )
        )

        sns.histplot(
            values,
            bins=np.arange(
                -1,
                1.05,
                0.05,
            ),
            ax=ax,
            color=color,
        )

        ax.scatter(
            mean_value,
            1,
            s=6,
            color="k",
            marker="v",
            zorder=2,
        )

        ax.text(
            mean_value,
            1.02,
            p_to_stars(
                p_value
            ),
            ha="center",
            va="bottom",
            fontsize=4,
            fontweight="bold",
            color="k",
            zorder=3,
        )

        ax.set_xlabel(
            xlabel
        )
        ax.set_ylabel(
            "Count"
        )
        ax.set_xlim(
            [
                -0.25,
                1.25,
            ]
        )
        ax.set_yticks(
            yticks
        )

        print(
            "-----------------------------"
        )

    @staticmethod
    def _print_exclusion_stats(
            panel_name,
            label,
            num_low_N,
    ):
        """Print low-N exclusion statistics for one panel."""
        low_n = np.asarray(
            num_low_N
        )
        models_with_low_n = np.count_nonzero(
            low_n
        )

        if models_with_low_n > 0:
            nonzero = low_n[
                low_n > 0
                ]
            mean_num_low_n = np.mean(
                nonzero
            )

            if models_with_low_n > 1:
                sem_num_low_n = (
                        np.std(
                            nonzero,
                            ddof=1,
                        )
                        / np.sqrt(
                    models_with_low_n
                )
                )
            else:
                sem_num_low_n = np.nan
        else:
            mean_num_low_n = 0.0
            sem_num_low_n = 0.0

        print(
            "Panel {}, stats on exclusions for {}".format(
                panel_name,
                label,
            )
        )
        print(
            "N models with stimulus combinations excluded from within model "
            "correlation summary due to low N: {}; "
            "mean +/- s.e.m. exclusions within those models: {} +/- {}".format(
                models_with_low_n,
                mean_num_low_n,
                sem_num_low_n,
            )
        )
        print(
            "-----------------------------"
        )
