"""Generate Extended Data Figure 7: HDDM parameter comparisons."""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from affective_task.utils import Constants, save_figure


class FigureS7:
    """Plot HDDM parameters across repeat and switch trial types."""

    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "dataset"
    hddm_stats_fn = "HDDM_group_results_vat_all-trials.csv"

    figsize = (9, 4)
    figdpi = 300

    BAR_ORDER = [
        "Gender Repeat",
        "Emotion Repeat",
        "Gender Switch",
        "Emotion Switch",
    ]
    BAR_COLORS = [
        Constants.COLOR_G2G,
        Constants.COLOR_E2E,
        Constants.COLOR_E2G,
        Constants.COLOR_G2E,
    ]

    GROUPS = [
        ("Gender", "Repeat"),
        ("Gender", "Switch"),
        ("Emotion", "Repeat"),
        ("Emotion", "Switch"),
    ]
    X_POS = {
        ("Gender", "Repeat"): 0,
        ("Emotion", "Repeat"): 1,
        ("Gender", "Switch"): 2,
        ("Emotion", "Switch"): 3,
    }

    def __init__(self, model_dir, save_dir, patients_id, rand_seed, n_boot):
        self.model_dir = model_dir
        self.save_dir = save_dir
        _ = patients_id, rand_seed, n_boot
        self.hddm_stats = None

    def make_figure(self):
        """Generate, save, and return Extended Data Figure 7."""
        print("Making Figure S7...")
        self._run_preprocessing()

        print("Stats for Figure S7")
        print("-------------------")
        fig = self._plot_figure_get_stats()

        save_figure(fig, self.save_dir, "FigS7")
        return fig

    def _run_preprocessing(self):
        """Load HDDM estimates and prepare categorical plotting labels."""
        path = self.dataset_dir / self.hddm_stats_fn
        if not path.is_file():
            raise FileNotFoundError("Missing required file: {}".format(path))

        df = pd.read_csv(path, index_col=0)
        df["task"] = df["trial"].str.split().str[0]
        df["condition"] = df["trial"].str.split().str[1]
        df["bar_label"] = df["task"] + " " + df["condition"]

        df["bar_label"] = pd.Categorical(
            df["bar_label"],
            categories=self.BAR_ORDER,
            ordered=True,
        )
        df["condition"] = pd.Categorical(
            df["condition"],
            categories=["Repeat", "Switch"],
            ordered=True,
        )
        df["task"] = pd.Categorical(
            df["task"],
            categories=["Gender", "Emotion"],
            ordered=True,
        )

        self.hddm_stats = df

    def _plot_figure_get_stats(self):
        """Assemble one panel for each HDDM parameter."""
        fig = plt.figure(
            constrained_layout=False,
            figsize=self.figsize,
            dpi=self.figdpi,
        )
        gs = fig.add_gridspec(7, 19)

        axes = [
            fig.add_subplot(gs[0:7, 0:5]),
            fig.add_subplot(gs[0:7, 7:12]),
            fig.add_subplot(gs[0:7, 14:19]),
        ]

        for ax, param in zip(axes, self.hddm_stats["param"].unique()):
            self._make_parameter_panel(ax, param)

        return fig

    def _make_parameter_panel(self, ax, param):
        """Plot one HDDM parameter with Holm-corrected paired comparisons."""
        sub = self.hddm_stats.loc[self.hddm_stats["param"] == param].copy()

        mean_df = (
            sub.groupby(
                ["subj_id", "task", "condition", "bar_label"]
            )["mean"]
            .mean()
            .reset_index()
        )

        group_df = (
            mean_df.groupby("bar_label")["mean"]
            .agg(["mean", "sem"])
            .reset_index()
            .rename(columns={"mean": "group_mean", "sem": "group_sem"})
        )

        sns.barplot(
            data=group_df,
            x="bar_label",
            y="group_mean",
            palette=self.BAR_COLORS,
            capsize=0.1,
            errcolor="black",
            errwidth=1.5,
            ax=ax,
        )

        for bar, (_, row) in zip(ax.patches, group_df.iterrows()):
            x = bar.get_x() + bar.get_width() / 2.0
            ax.errorbar(
                x,
                bar.get_height(),
                yerr=row["group_sem"],
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=6,
                capthick=1.0,
            )

        sns.stripplot(
            data=mean_df,
            x="bar_label",
            y="mean",
            palette=self.BAR_COLORS,
            alpha=0.7,
            size=4,
            edgecolor="black",
            linewidth=0.2,
            jitter=False,
            ax=ax,
        )

        results = self._paired_comparisons(mean_df)

        if results:
            adjusted = self._holm_bonferroni(
                [result["p_raw"] for result in results]
            )
            for result, p_adj in zip(results, adjusted):
                result["p_adj"] = float(p_adj)
                result["stars"] = self._stars_from_adjusted_p(p_adj)

            self._print_stat_results(param, results)
            self._add_comparison_brackets(ax, group_df, results)
        else:
            print("{}: no valid paired comparisons available.".format(param))

        ax.set_title(str(param), fontsize=8)
        ax.set_xlabel("Trial type")
        ax.set_ylabel(str(param))
        ax.set_xticklabels(["G2G", "E2E", "E2G", "G2E"], fontsize=8)

        if ax.legend_:
            ax.legend_.remove()

    def _paired_comparisons(self, mean_df):
        """Run all valid paired trial-type comparisons for one parameter."""
        wide = mean_df.pivot_table(
            index="subj_id",
            columns=["task", "condition"],
            values="mean",
            aggfunc="mean",
        )

        pairs = sorted(
            itertools.combinations(self.GROUPS, 2),
            key=lambda pair: abs(
                self.X_POS[pair[1]] - self.X_POS[pair[0]]
            ),
        )

        results = []
        for group_1, group_2 in pairs:
            if group_1 not in wide.columns or group_2 not in wide.columns:
                continue

            paired = wide.loc[:, [group_1, group_2]].dropna()
            if len(paired) < 2:
                continue

            t_stat, p_raw = stats.ttest_rel(
                paired[group_1].to_numpy(dtype=float),
                paired[group_2].to_numpy(dtype=float),
            )
            if not np.isfinite(p_raw):
                continue

            results.append(
                {
                    "g1": group_1,
                    "g2": group_2,
                    "n": len(paired),
                    "t": float(t_stat),
                    "p_raw": float(p_raw),
                }
            )

        return results

    def _add_comparison_brackets(self, ax, group_df, results):
        """Add non-overlapping brackets using Holm-adjusted significance labels."""
        ymax = float(
            (
                    group_df["group_mean"]
                    + group_df["group_sem"].fillna(0)
            ).max()
        )
        y_range = float(ax.get_ylim()[1] - ax.get_ylim()[0])
        if y_range <= 0:
            y_range = max(1.0, abs(ymax))

        base_y = ymax + 0.08 * y_range
        step_y = 0.07 * y_range
        bracket_h = 0.015 * y_range
        used_levels = []

        for result in results:
            x1 = self.X_POS[result["g1"]]
            x2 = self.X_POS[result["g2"]]

            level = 0
            while True:
                y = base_y + level * step_y
                conflict = any(
                    abs(used_y - y) < 0.5 * step_y
                    and self._overlaps(x1, x2, used_x1, used_x2)
                    for used_x1, used_x2, used_y in used_levels
                )
                if not conflict:
                    break
                level += 1

            self._add_bracket(
                ax,
                x1,
                x2,
                y,
                bracket_h,
                result["stars"],
                fontsize=4,
            )
            used_levels.append((x1, x2, y))

        top_y = (
                max(y + bracket_h for _, _, y in used_levels)
                + 0.06 * y_range
        )
        ax.set_ylim(
            ax.get_ylim()[0],
            max(ax.get_ylim()[1], top_y),
        )

    @staticmethod
    def _print_stat_results(param, results):
        """Print raw and Holm-adjusted paired t-test results."""
        print(
            "{}: Holm--Bonferroni correction across {} paired t-tests "
            "(all trial-type comparisons within this parameter plot)".format(
                param,
                len(results),
            )
        )

        for result in results:
            g1 = result["g1"]
            g2 = result["g2"]
            print(
                "{}: {} {} vs {} {} -> n={}, t={:.3f}, "
                "p_raw={:.4g}, p_Holm={:.4g} ({})".format(
                    param,
                    g1[0],
                    g1[1],
                    g2[0],
                    g2[1],
                    result["n"],
                    result["t"],
                    result["p_raw"],
                    result["p_adj"],
                    result["stars"],
                )
            )

        print("--------------------------------------------")

    @staticmethod
    def _add_bracket(ax, x1, x2, y, h, text, fontsize=12):
        """Draw one significance bracket."""
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + h, y + h, y],
            color="black",
            lw=1.2,
        )
        ax.text(
            (x1 + x2) / 2.0,
            y + h,
            text,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
        )

    @staticmethod
    def _overlaps(a1, a2, b1, b2):
        """Return whether two bracket x-ranges overlap."""
        lo1, hi1 = sorted([a1, a2])
        lo2, hi2 = sorted([b1, b2])
        return not (hi1 < lo2 or hi2 < lo1)

    @staticmethod
    def _holm_bonferroni(p_values):
        """Return Holm--Bonferroni-adjusted p-values in input order."""
        p_values = np.asarray(p_values, dtype=float)

        if p_values.ndim != 1:
            raise ValueError("p_values must be one-dimensional.")
        if len(p_values) == 0:
            return np.array([], dtype=float)
        if np.any(~np.isfinite(p_values)):
            raise ValueError(
                "All p-values must be finite for Holm correction."
            )

        m = len(p_values)
        order = np.argsort(p_values)
        adjusted = np.empty(m, dtype=float)
        running_max = 0.0

        for rank, index in enumerate(order):
            candidate = (m - rank) * p_values[index]
            running_max = max(running_max, candidate)
            adjusted[index] = min(running_max, 1.0)

        return adjusted

    @staticmethod
    def _stars_from_adjusted_p(p_adj):
        """Convert a Holm-adjusted p-value to significance notation."""
        if p_adj < 1e-4:
            return "****"
        if p_adj < 1e-3:
            return "***"
        if p_adj < 1e-2:
            return "**"
        if p_adj < 0.05:
            return "*"
        return "n.s."
