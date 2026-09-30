"""Generate Figure 2: behavioral validation of GeoDynFormer."""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from affective_task.utils import (
    Constants,
    p_to_stars,
    plot_scatter,
    save_figure,
)
from affective_task.visualization import PlotRTs


class Figure2:
    """Plot participant/model RT distributions and switch-cost analyses."""

    analysis_dir = "model_analysis"
    summary_fn = "summary.pkl"
    stats_fn = "holdout_outputs_01SD.pkl"

    line_ext = 10
    figsize = (12, 6)
    figdpi = 300

    # Exemplars used for the RT-distribution panels.
    exemplar_ids = {
        "37": "A1",
        "71_2": "A2",
        "35": "B1",
        "66_1": "B2",
        "27": "C1",
        "57": "C2",
    }

    def __init__(self, model_dir, save_dir, patients_id, rand_seed, n_boot):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.patients_id = patients_id
        self.rng = np.random.default_rng(rand_seed)
        self.n_boot = n_boot
        self.alpha = 0.05

        # Only mean RT is required by the active participant-vs-model scatter.
        self.group_stats = {
            "u_mean_rt": [],
            "m_mean_rt": [],
        }

        self.exemplars = {}
        self.u_switch_costs_stats = {}
        self.m_switch_costs_stats = {}

    def make_figure(self):
        """Generate and save Figure 2."""
        print("Making Figure 2...")
        self._run_preprocessing()

        print("Stats for Figure 2")
        print("------------------")

        fig = self._plot_figure_get_stats()
        save_figure(fig, self.save_dir, "Fig2")

    def _run_preprocessing(self):
        """Load participant-level model outputs and summary statistics."""
        for patient_id in self.patients_id:
            subject_dir = os.path.join(
                self.model_dir,
                "s{}".format(patient_id),
            )

            stats_path = os.path.join(
                subject_dir,
                self.analysis_dir,
                self.stats_fn,
            )
            with open(stats_path, "rb") as path:
                expt_stats = pickle.load(path)

            for key in self.group_stats:
                self.group_stats[key].append(expt_stats.summary_stats[key])

            if patient_id in self.exemplar_ids:
                exemplar_key = self.exemplar_ids[patient_id]
                self.exemplars[exemplar_key] = expt_stats

            summary_path = os.path.join(
                subject_dir,
                self.analysis_dir,
                self.summary_fn,
            )
            with open(summary_path, "rb") as path:
                summary_stats = pickle.load(path)

            primary_summary = summary_stats["01"]

            self.u_switch_costs_stats[patient_id] = {
                "E2G": primary_summary["u_switch_cost_trial_type_2"],
                "G2E": primary_summary["u_switch_cost_trial_type_3"],
            }
            self.m_switch_costs_stats[patient_id] = {
                "E2G": primary_summary["m_switch_cost_trial_type_2"],
                "G2E": primary_summary["m_switch_cost_trial_type_3"],
            }

        self.u_switch_costs_df = self._make_switch_cost_df(
            self.u_switch_costs_stats
        )
        self.m_switch_costs_df = self._make_switch_cost_df(
            self.m_switch_costs_stats
        )

    @staticmethod
    def _make_switch_cost_df(switch_cost_stats):
        """Convert subject-level switch costs to long-form plotting data."""
        df = (
            pd.DataFrame(switch_cost_stats)
            .T
            .reset_index()
            .rename(columns={"index": "subj_id"})
            .melt(
                id_vars="subj_id",
                var_name="transition",
                value_name="switch_cost",
            )
        )
        df["transition"] = pd.Categorical(
            df["transition"],
            categories=["E2G", "G2E"],
            ordered=True,
        )
        return df

    def _plot_figure_get_stats(self):
        """Assemble Figure 2 and print the associated statistics."""
        fig = plt.figure(
            constrained_layout=False,
            figsize=self.figsize,
            dpi=self.figdpi,
        )
        gs = fig.add_gridspec(14, 28)

        # Exemplar RT-distribution panels.
        exemplar_axes = {
            "A1": fig.add_subplot(gs[0:6, 0:3]),
            "A2": fig.add_subplot(gs[0:6, 5:8]),
            "B1": fig.add_subplot(gs[0:6, 10:13]),
            "B2": fig.add_subplot(gs[0:6, 15:18]),
            "C1": fig.add_subplot(gs[0:6, 20:23]),
            "C2": fig.add_subplot(gs[0:6, 25:28]),
        }
        self._make_panels_ABC(exemplar_axes)

        # Participant vs. model mean RT.
        mean_rt_params = {
            "ax_lims": [1000, 2100],
            "metric": "mean_rt",
            "label": "mean RT",
        }
        mean_rt_ax = fig.add_subplot(gs[8:14, 0:6])
        plot_scatter(
            self.group_stats,
            mean_rt_params,
            mean_rt_ax,
            self.line_ext,
            self.rng,
            n_boot=self.n_boot,
            alpha=self.alpha,
        )

        # Participant and model switch costs.
        participant_sc_ax = fig.add_subplot(gs[8:14, 8:14])
        self._make_panel_E1(
            self.u_switch_costs_df,
            participant_sc_ax,
            range_ylim=[-500, 1800],
        )

        model_sc_ax = fig.add_subplot(gs[8:14, 16:22])
        self._make_panel_E2(
            self.m_switch_costs_df,
            model_sc_ax,
            range_ylim=[-500, 1800],
        )

        return fig

    def _make_panels_ABC(self, axes):
        """Plot exemplar participant/model RT distributions."""
        for key, ax in axes.items():
            expt_stats = self.exemplars[key]

            if key.startswith("A"):
                plot_type = "all"
            elif key.startswith("B"):
                plot_type = "Trial_type_2"
            elif key.startswith("C"):
                plot_type = "Trial_type_3"
            else:
                raise KeyError(key)

            plotter = PlotRTs(expt_stats)
            ax = plotter.plot_rt_dists(ax, plot_type)

            if key == "A1":
                ax.get_legend().set_title(None)
                handles, _ = ax.get_legend_handles_labels()
                ax.legend(
                    handles=handles,
                    labels=["Participant", "Model"],
                    bbox_to_anchor=(1.0, 1.4),
                    loc="upper right",
                    borderaxespad=0.0,
                )
                ax.get_legend().get_frame().set_linewidth(0.0)
                ax.set_ylabel("RT (ms)")
            elif key == "A2":
                ax.get_legend().remove()
            elif key.startswith("B"):
                ax.get_legend().remove()
                ax.set_title("Gender Trials")
            elif key.startswith("C"):
                ax.get_legend().remove()
                ax.set_title("Emotion Trials")

    def _make_panel_E1(self, df, ax, range_ylim=None):
        """Plot participant E2G and G2E switch costs."""
        self._make_switch_cost_panel(
            df,
            ax,
            panel_name="Participant switch cost",
            ylabel="Participant switch cost",
            range_ylim=range_ylim,
        )

    def _make_panel_E2(self, df, ax, range_ylim=None):
        """Plot model E2G and G2E switch costs."""
        self._make_switch_cost_panel(
            df,
            ax,
            panel_name="Model switch cost",
            ylabel="Model switch cost",
            range_ylim=range_ylim,
        )

    def _make_switch_cost_panel(
            self,
            df,
            ax,
            panel_name,
            ylabel,
            range_ylim=None,
    ):
        """Plot paired switch costs and report one-sample/paired t-tests."""
        df = df.copy()

        order = ["E2G", "G2E"]
        xpos = {transition: i for i, transition in enumerate(order)}
        palette = [
            Constants.COLOR_E2G,
            Constants.COLOR_G2E,
        ]

        # Subject-level aggregation.
        df_subj = (
            df.groupby(
                ["subj_id", "transition"],
                as_index=False,
            )["switch_cost"]
            .mean()
        )

        summary = (
            df_subj.groupby("transition")["switch_cost"]
            .agg(
                mean_sc="mean",
                sd_sc="std",
                n="count",
            )
            .reset_index()
        )
        summary["sem_sc"] = summary["sd_sc"] / np.sqrt(summary["n"])

        print("\n{} summary statistics:".format(panel_name))
        print(summary)

        # One-sample t-tests against zero.
        print("\n{} one-sample t-tests against 0:".format(panel_name))
        for transition in order:
            values = df_subj.loc[
                df_subj["transition"] == transition,
                "switch_cost",
            ].dropna()

            t_stat, p_val = stats.ttest_1samp(values, 0.0)
            mean_val = values.mean()
            sem_val = values.std(ddof=1) / np.sqrt(len(values))

            print(
                "{}: mean ± SEM = {:.4f} ± {:.4f}, "
                "t({}) = {:.4f}, p = {:.6g}, N = {}".format(
                    transition,
                    mean_val,
                    sem_val,
                    len(values) - 1,
                    t_stat,
                    p_val,
                    len(values),
                )
            )

        # Paired t-test: E2G vs. G2E.
        wide = df_subj.pivot_table(
            index="subj_id",
            columns="transition",
            values="switch_cost",
        )
        wide_pair = wide.dropna(subset=order)

        e2g_values = wide_pair["E2G"]
        g2e_values = wide_pair["G2E"]
        t_stat_pair, p_val_pair = stats.ttest_rel(
            e2g_values,
            g2e_values,
        )

        difference = g2e_values - e2g_values
        mean_diff = difference.mean()
        sem_diff = difference.std(ddof=1) / np.sqrt(len(difference))

        print("\n{} paired t-test:".format(panel_name))
        print(
            "G2E - E2G: mean difference ± SEM = {:.4f} ± {:.4f}, "
            "t({}) = {:.4f}, p = {:.6g}, N = {}".format(
                mean_diff,
                sem_diff,
                len(difference) - 1,
                t_stat_pair,
                p_val_pair,
                len(difference),
            )
        )

        # Boxplot and participant-level paired observations.
        sns.boxplot(
            data=df,
            x="transition",
            y="switch_cost",
            order=order,
            palette=palette,
            showcaps=True,
            fliersize=0,
            width=0.7,
            linewidth=0.5,
            ax=ax,
        )
        sns.stripplot(
            data=df,
            x="transition",
            y="switch_cost",
            order=order,
            palette=palette,
            alpha=0.7,
            size=4,
            jitter=False,
            zorder=10,
            linewidth=0.2,
            ax=ax,
        )

        for _, group in df_subj.groupby("subj_id", sort=False):
            if set(group["transition"]) >= set(order):
                xs = [xpos[transition] for transition in order]
                ys = [
                    group.loc[
                        group["transition"] == transition,
                        "switch_cost",
                    ].iloc[0]
                    for transition in order
                ]
                ax.plot(
                    xs,
                    ys,
                    color="gray",
                    alpha=0.6,
                    lw=0.8,
                    zorder=2,
                )

        y_max_data = float(df["switch_cost"].max())

        if range_ylim is not None:
            ax.set_ylim(range_ylim)
        else:
            y_min_data = float(df["switch_cost"].min())
            y_min = min(y_min_data, 0.0)
            y_span = (
                y_max_data - y_min
                if (y_max_data - y_min) > 1e-9
                else 1.0
            )
            y_top = y_max_data + 0.3 * y_span
            ax.set_ylim(
                y_min - 0.1 * y_span,
                y_top,
            )

        # Place significance annotations relative to the final axis limits.
        y_min_ax, y_max_ax = ax.get_ylim()
        y_span_ax = y_max_ax - y_min_ax

        within_y = y_max_data + 0.01 * y_span_ax
        bracket_y = y_max_data + 0.09 * y_span_ax
        bracket_h = 0.015 * y_span_ax
        text_pad = 0.001 * y_span_ax

        max_text_y = y_max_ax - 0.01 * y_span_ax
        within_y = min(within_y, max_text_y)
        bracket_y = min(
            bracket_y,
            max_text_y - (bracket_h + text_pad),
        )

        # One-sample significance markers.
        for i, transition in enumerate(order):
            values = df_subj.loc[
                df_subj["transition"] == transition,
                "switch_cost",
            ].dropna()
            _, p_val = stats.ttest_1samp(values, 0.0)

            ax.text(
                i,
                within_y,
                p_to_stars(p_val),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                clip_on=True,
                zorder=30,
            )

        # Paired-comparison significance bracket.
        self._add_sig_bracket_inside(
            ax,
            xpos["E2G"],
            xpos["G2E"],
            y=bracket_y,
            h=bracket_h,
            text=p_to_stars(p_val_pair),
            text_pad=text_pad,
            lw=1.0,
            fontsize=8,
        )

        ax.set_xlabel("Task transition")
        ax.set_ylabel(ylabel)

    @staticmethod
    def _add_sig_bracket_inside(
            ax,
            x1,
            x2,
            y,
            h,
            text="*",
            text_pad=0.0,
            lw=1.5,
            fontsize=14,
    ):
        """Draw a significance bracket inside the current axis limits."""
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + h, y + h, y],
            color="black",
            lw=lw,
            clip_on=True,
            zorder=25,
        )
        ax.text(
            (x1 + x2) / 2.0,
            y + h + text_pad,
            text,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
            clip_on=True,
            zorder=26,
        )
