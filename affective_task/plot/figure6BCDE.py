"""Generate Figure 6b-e: switch-cost geometry across participants."""

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import PearsonRConstantInputWarning
from statsmodels.stats.multitest import multipletests

from affective_task.model_analysis import LatentSeparation
from affective_task.utils import (
    Constants,
    bootstrap_regression_band,
    p_to_stars,
    pearson_bootstrap,
    save_figure,
)


class Figure6BCDE:
    """Plot switch-cost correlations and E2G/G2E geometric comparisons."""

    analysis_dir = "model_analysis"
    stats_fn = "holdout_outputs_01SD.pkl"

    figsize = (9, 9)
    figdpi = 300

    def __init__(self, model_dir, save_dir, patients_id, rand_seed, n_boot):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.patients_id = patients_id
        self.rng = np.random.default_rng(rand_seed)
        self.n_boot = n_boot
        self.alpha = 0.05

        # Pearson bootstrap resamples can occasionally contain constant input.
        warnings.simplefilter("ignore", PearsonRConstantInputWarning)

        self.e2g_distances = []
        self.e2g_switch_costs = []
        self.g2e_distances = []
        self.g2e_switch_costs = []

        self.distance_by_transition = {}
        self.switch_cost_by_transition = {}

    def make_figure(self):
        """Generate and save Figure 6b-e."""
        print("Making Figure 6b-e...")
        self._run_preprocessing()

        print("Stats for Figure 6b-e")
        print("---------------------")

        fig = self._plot_figure_get_stats()
        save_figure(fig, self.save_dir, "Fig6bcde")
        return fig

    def _run_preprocessing(self):
        """Compute participant-level E2G/G2E distances and switch costs."""
        for patient_id in self.patients_id:
            stats_path = os.path.join(
                self.model_dir,
                "s{}".format(patient_id),
                self.analysis_dir,
                self.stats_fn,
            )
            with open(stats_path, "rb") as path:
                expt_stats = pickle.load(path)

            latent_separation = LatentSeparation(expt_stats)

            e2g_stats = latent_separation.analyze_latent_dist_switch(
                current_task=0
            )
            g2e_stats = latent_separation.analyze_latent_dist_switch(
                current_task=1
            )

            e2g_distance = e2g_stats["euc_dist"]
            g2e_distance = g2e_stats["euc_dist"]
            e2g_switch_cost = expt_stats.summary_stats[
                "u_switch_cost_trial_type_2"
            ]
            g2e_switch_cost = expt_stats.summary_stats[
                "u_switch_cost_trial_type_3"
            ]

            self.e2g_distances.append(e2g_distance)
            self.e2g_switch_costs.append(e2g_switch_cost)
            self.g2e_distances.append(g2e_distance)
            self.g2e_switch_costs.append(g2e_switch_cost)

            self.distance_by_transition[patient_id] = {
                "E2G": e2g_distance,
                "G2E": g2e_distance,
            }
            self.switch_cost_by_transition[patient_id] = {
                "E2G": e2g_switch_cost,
                "G2E": g2e_switch_cost,
            }

        self.distance_df = self._make_transition_dataframe(
            self.distance_by_transition,
            value_name="normed_dist",
        )
        self.switch_cost_df = self._make_transition_dataframe(
            self.switch_cost_by_transition,
            value_name="switch_cost",
        )

        self.switch_cost_distance_df = pd.merge(
            self.distance_df[
                ["subj_id", "transition", "normed_dist"]
            ],
            self.switch_cost_df[
                ["subj_id", "transition", "switch_cost"]
            ],
            on=["subj_id", "transition"],
            how="inner",
        )

    @staticmethod
    def _make_transition_dataframe(values_by_subject, value_name):
        """Convert subject-level E2G/G2E values to long-form data."""
        df = (
            pd.DataFrame(values_by_subject)
            .T
            .reset_index()
            .rename(columns={"index": "subj_id"})
            .melt(
                id_vars="subj_id",
                var_name="transition",
                value_name=value_name,
            )
        )
        df["transition"] = pd.Categorical(
            df["transition"],
            categories=["E2G", "G2E"],
            ordered=True,
        )
        return df

    def _plot_figure_get_stats(self):
        """Assemble Figure 6b-e and print the associated statistics."""
        fig = plt.figure(
            constrained_layout=False,
            figsize=self.figsize,
            dpi=self.figdpi,
        )
        gs = fig.add_gridspec(9, 9)

        # Figure 6b: E2G distance vs. participant switch cost.
        ax_e2g = fig.add_subplot(gs[0:4, 0:4])
        stats_e2g = self._make_panel_C(
            ax_e2g,
            self.e2g_distances,
            self.e2g_switch_costs,
            trial_type=2,
            data_type="participant",
            color=Constants.COLOR_E2G,
            range_xlim=[-450, 1450],
            range_ylim=[20, 90],
        )

        # Figure 6c: G2E distance vs. participant switch cost.
        ax_g2e = fig.add_subplot(gs[0:4, 5:9])
        stats_g2e = self._make_panel_C(
            ax_g2e,
            self.g2e_distances,
            self.g2e_switch_costs,
            trial_type=3,
            data_type="participant",
            color=Constants.COLOR_G2E,
            range_xlim=[-450, 1450],
            range_ylim=[20, 90],
        )

        raw_pvalues = [
            stats_e2g["p"],
            stats_g2e["p"],
        ]
        reject, corrected_pvalues, _, _ = multipletests(
            raw_pvalues,
            alpha=0.05,
            method="holm",
        )

        print("Raw p-values:", raw_pvalues)
        print("Holm-corrected p-values:", corrected_pvalues)
        print("Significant after correction:", reject)

        # Figure 6d: paired E2G/G2E onset-point distances.
        ax_distance = fig.add_subplot(gs[5:9, 0:4])
        self._make_panel_D(
            self.distance_df,
            ax_distance,
        )

        # Figure 6e: repeated-measures distance/switch-cost association.
        ax_rmcorr = fig.add_subplot(gs[5:9, 5:9])
        self._make_panel_F(
            ax_rmcorr,
            self.switch_cost_distance_df,
        )

        return fig

    def _make_panel_C(
            self,
            ax,
            distances,
            switch_costs,
            trial_type,
            color,
            line_ext=10,
            data_type="model",
            plot_boot_band=True,
            n_boot_band=10000,
            ci=95,
            seed=0,
            range_xlim=None,
            range_ylim=None,
    ):
        """Plot onset-point distance against switch cost for one transition."""
        transition = "E2G" if trial_type == 2 else "G2E"

        distances = np.asarray(distances)
        switch_costs = np.asarray(switch_costs)

        plot_x = np.array(
            [
                np.min(switch_costs) - line_ext,
                np.max(switch_costs) + line_ext,
            ]
        )
        slope, intercept = np.polyfit(
            switch_costs,
            distances,
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
            switch_costs,
            distances,
            s=4,
            marker="o",
            zorder=1,
            color=color,
        )

        ax.set_xlabel(
            "{} switch cost (ms) for {}".format(
                data_type,
                transition,
            )
        )
        ax.set_ylabel(
            "Euclidean distance between\ntask onset points (a.u.)"
        )

        print(
            "Panel C stats: Euclidean dist. vs. "
            "{} switch costs for {}".format(
                data_type,
                transition,
            )
        )

        r_value, p_value, ci_low, ci_high = pearson_bootstrap(
            switch_costs,
            distances,
            self.rng,
            n_boot=self.n_boot,
            alpha=self.alpha,
        )

        print(
            "r = {}, 95% CI: ({}, {}), p = {:0.2e} ({})".format(
                round(r_value, 2),
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

        mean_distance = np.mean(distances)
        sem_distance = np.std(distances) / np.sqrt(len(distances))
        print(
            "Mean +/- s.e.m. centroid distance: {} +/- {}".format(
                mean_distance,
                sem_distance,
            )
        )

        mean_switch_cost = np.mean(switch_costs)
        sem_switch_cost = (
                np.std(switch_costs)
                / np.sqrt(len(switch_costs))
        )
        print(
            "Mean +/- s.e.m. switch cost (ms): {} +/- {}".format(
                mean_switch_cost,
                sem_switch_cost,
            )
        )
        print("-----------------------------")

        if plot_boot_band:
            y_hat, y_low, y_high = bootstrap_regression_band(
                switch_costs,
                distances,
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

        if range_xlim is not None:
            ax.set_xlim(range_xlim)
        if range_ylim is not None:
            ax.set_ylim(range_ylim)

        return {
            "r": r_value,
            "p": p_value,
            "ci_lo": ci_low,
            "ci_hi": ci_high,
        }

    def _make_panel_D(self, df, ax):
        """Plot paired E2G/G2E Euclidean onset-point distances."""
        df = df.copy()

        order = ["E2G", "G2E"]
        xpos = {
            transition: i
            for i, transition in enumerate(order)
        }
        palette = [
            Constants.COLOR_E2G,
            Constants.COLOR_G2E,
        ]

        df_subject = (
            df.groupby(
                ["subj_id", "transition"],
                as_index=False,
            )["normed_dist"]
            .mean()
        )

        summary = (
            df_subject.groupby("transition")["normed_dist"]
            .agg(
                mean_sc="mean",
                sd_sc="std",
                n="count",
            )
            .reset_index()
        )
        summary["sem_sc"] = (
                summary["sd_sc"]
                / np.sqrt(summary["n"])
        )

        print("Panel D summary statistics:")
        print(summary)

        sns.boxplot(
            data=df,
            x="transition",
            y="normed_dist",
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
            y="normed_dist",
            order=order,
            palette=palette,
            alpha=0.7,
            size=4,
            jitter=False,
            zorder=10,
            linewidth=0.2,
            ax=ax,
        )

        for _, group in df.groupby(
                "subj_id",
                sort=False,
        ):
            if set(group["transition"]) >= set(order):
                xs = [
                    xpos[transition]
                    for transition in order
                ]
                ys = [
                    group.loc[
                        group["transition"] == transition,
                        "normed_dist",
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

        y_min_data = float(
            df["normed_dist"].min()
        )
        y_max_data = float(
            df["normed_dist"].max()
        )
        y_span = (
            y_max_data - y_min_data
            if y_max_data - y_min_data > 1e-9
            else 1.0
        )

        y_top = y_max_data + 0.3 * y_span
        ax.set_ylim(
            y_min_data - 0.1 * y_span,
            y_top,
        )

        y_min_ax, y_max_ax = ax.get_ylim()
        y_span_ax = y_max_ax - y_min_ax

        within_y = y_max_data + 0.01 * y_span_ax
        bracket_y = y_max_data + 0.09 * y_span_ax
        bracket_h = 0.015 * y_span_ax
        text_pad = 0.001 * y_span_ax

        max_text_y = y_top - 0.01 * y_span_ax
        within_y = min(
            within_y,
            max_text_y,
        )
        bracket_y = min(
            bracket_y,
            max_text_y - (bracket_h + text_pad),
        )

        print("Panel D one-sample t-tests against zero:")
        for i, transition in enumerate(order):
            values = df.loc[
                df["transition"] == transition,
                "normed_dist",
            ].dropna()

            t_stat, p_value = stats.ttest_1samp(
                values,
                0.0,
            )

            print(
                "{}: one-sample t-test vs. 0, "
                "t({}) = {}, p = {}, N = {}, "
                "mean +/- s.e.m. = {} +/- {}".format(
                    transition,
                    len(values) - 1,
                    t_stat,
                    p_value,
                    len(values),
                    values.mean(),
                    stats.sem(
                        values,
                        nan_policy="omit",
                    ),
                )
            )

            ax.text(
                i,
                within_y,
                p_to_stars(p_value),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                clip_on=True,
                zorder=30,
            )

        wide = df.pivot_table(
            index="subj_id",
            columns="transition",
            values="normed_dist",
            aggfunc="mean",
        )
        paired = wide[order].dropna()

        e2g_values = paired["E2G"].values
        g2e_values = paired["G2E"].values

        t_stat, p_value = stats.ttest_rel(
            e2g_values,
            g2e_values,
            nan_policy="omit",
        )
        difference = (
                e2g_values - g2e_values
        )

        print("Panel D paired t-test:")
        print(
            "E2G vs. G2E: paired t-test, "
            "t({}) = {}, p = {}, N = {}".format(
                len(paired) - 1,
                t_stat,
                p_value,
                len(paired),
            )
        )
        print(
            "Mean paired difference (E2G - G2E) +/- s.e.m. = "
            "{} +/- {}".format(
                np.mean(difference),
                stats.sem(
                    difference,
                    nan_policy="omit",
                ),
            )
        )

        self._add_sig_bracket_inside(
            ax,
            xpos["E2G"],
            xpos["G2E"],
            y=bracket_y,
            h=bracket_h,
            text=p_to_stars(p_value),
            text_pad=text_pad,
            lw=1.0,
            fontsize=8,
        )

        ax.set_xlabel("Task transition")
        ax.set_ylabel("Euclidean Distance")

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

    def _make_panel_F(
            self,
            ax,
            df,
            x_label="Euclidean distance",
            y_label="Switch cost",
            title="Repeated-measures correlation (rmcorr)",
            annotate=False,
            show_points=True,
            show_lines=True,
    ):
        """Plot and report the repeated-measures distance/switch-cost correlation."""
        import pingouin as pg
        from matplotlib.collections import PathCollection

        result = pg.rm_corr(
            data=df,
            x="normed_dist",
            y="switch_cost",
            subject="subj_id",
        )
        if result is None or len(result) == 0:
            raise RuntimeError(
                "pingouin.rm_corr returned no results. "
                "Check that each subject has at least 2 non-NaN observations."
            )

        result_row = result.iloc[0]
        r_value = float(
            result_row["r"]
        )
        p_value = float(
            result_row["pval"]
        )

        ci_cell = result_row.get(
            "CI95%",
            None,
        )
        if ci_cell is None:
            ci_low = float(
                result_row.get(
                    "CI95%[0]",
                    np.nan,
                )
            )
            ci_high = float(
                result_row.get(
                    "CI95%[1]",
                    np.nan,
                )
            )
        else:
            ci_low = float(ci_cell[0])
            ci_high = float(ci_cell[1])

        # Pingouin returns a FacetGrid. Build it off-screen and copy its
        # subject-specific artists into the axis supplied by this figure.
        custom_palette = (
                                 list(plt.get_cmap("tab20").colors)[:12]
                                 + sns.color_palette("Dark2", 8)
                                 + sns.color_palette("Set1", 9)
                         )[:22]

        grid = pg.plot_rm_corr(
            data=df,
            x="normed_dist",
            y="switch_cost",
            subject="subj_id",
            kwargs_facetgrid={
                "palette": custom_palette,
            },
            kwargs_scatter={
                "s": 10,
                "alpha": 0.85,
            },
            kwargs_line={
                "linewidth": 0.6,
                "alpha": 0.8,
            },
        )

        source_ax = grid.ax
        source_fig = grid.fig

        if show_lines:
            for line in list(source_ax.lines):
                ax.plot(
                    line.get_xdata(),
                    line.get_ydata(),
                    color=line.get_color(),
                    linewidth=line.get_linewidth(),
                    linestyle=line.get_linestyle(),
                    alpha=(
                        line.get_alpha()
                        if line.get_alpha() is not None
                        else 1.0
                    ),
                    solid_capstyle=line.get_solid_capstyle(),
                    zorder=line.get_zorder(),
                )

        if show_points:
            for collection in list(
                    source_ax.collections
            ):
                if not isinstance(
                        collection,
                        PathCollection,
                ):
                    continue

                offsets = collection.get_offsets()
                if offsets is None or len(offsets) == 0:
                    continue

                sizes = collection.get_sizes()
                facecolors = collection.get_facecolors()
                edgecolors = collection.get_edgecolors()
                linewidths = collection.get_linewidths()
                alpha = collection.get_alpha()

                ax.scatter(
                    offsets[:, 0],
                    offsets[:, 1],
                    s=(
                        sizes
                        if len(sizes)
                        else None
                    ),
                    facecolors=(
                        facecolors
                        if len(facecolors)
                        else None
                    ),
                    edgecolors=(
                        edgecolors
                        if len(edgecolors)
                        else None
                    ),
                    linewidths=(
                        linewidths
                        if len(linewidths)
                        else None
                    ),
                    alpha=(
                        alpha
                        if alpha is not None
                        else 1.0
                    ),
                    zorder=collection.get_zorder(),
                )

        ax.set_xlim(source_ax.get_xlim())
        ax.set_ylim(source_ax.get_ylim())
        plt.close(source_fig)

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        stat_text = (
            "rmcorr r = {:.3f}\n"
            "95% CI = ({:.3f}, {:.3f})\n"
            "p = {:.3e}\n"
            "N = {}, subjects = {}".format(
                r_value,
                ci_low,
                ci_high,
                p_value,
                len(df),
                df["subj_id"].nunique(),
            )
        )

        print("{}:\n{}".format(title, stat_text))
        print("--------------------------------------------------")

        if annotate:
            ax.text(
                0.02,
                0.98,
                stat_text,
                transform=ax.transAxes,
                va="top",
                fontsize=10,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "alpha": 0.85,
                    "linewidth": 0,
                },
            )
