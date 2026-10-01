"""Generate Extended Data Figure 1: Transformer vs RNN architecture comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from affective_task.utils import Constants


class FigureS1:
    """Plot the Transformer-versus-RNN architecture-ablation results."""

    project_root = Path(__file__).resolve().parents[2]
    input_dir = project_root / "dataset"

    paired_summary_fn = "paired_subject_summary_arch_comparison.csv"
    statistics_fn = "paired_statistics_arch_comparison.csv"

    figsize = (11.4, 3.9)
    figdpi = 300

    NEUTRAL_EDGE = "#303030"
    NEUTRAL_FILL = "#BDBDBD"
    PAIR_LINE_COLOR = "#A6A6A6"

    TRIAL_COLORS = {
        "G2G": Constants.COLOR_G2G,
        "E2E": Constants.COLOR_E2E,
        "E2G": Constants.COLOR_E2G,
        "G2E": Constants.COLOR_G2E,
    }

    def __init__(self, model_dir, save_dir, patients_id, rand_seed, n_boot):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.patients_id = patients_id

        # Retain the common figure-class constructor signature.
        _ = rand_seed, n_boot

        self.paired = None
        self.statistics_df = None

    def make_figure(self):
        """Generate, save, display, and return Extended Data Figure 1."""
        print("Making Figure S1...")
        self._run_preprocessing()

        print("Stats for Figure S1")
        print("-------------------")
        fig = self._plot_figure_get_stats()

        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        png_path = save_dir / "FigS1.png"
        svg_path = save_dir / "FigS1.svg"

        fig.savefig(
            png_path,
            dpi=self.figdpi,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )
        fig.savefig(
            svg_path,
            format="svg",
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )

        print("Saved:")
        print("- {}".format(png_path))
        print("- {}".format(svg_path))

        plt.show()
        return fig

    def _run_preprocessing(self):
        """Load precomputed architecture-comparison tables."""
        paired_path = self.input_dir / self.paired_summary_fn
        stats_path = self.input_dir / self.statistics_fn

        for path in (paired_path, stats_path):
            if not path.is_file():
                raise FileNotFoundError("Missing required file: {}".format(path))

        self.paired = pd.read_csv(paired_path)
        self.statistics_df = pd.read_csv(stats_path)

        paired_columns = [
            "selected_mean_rt_mae_rnn",
            "selected_mean_rt_mae_transformer",
            "selected_repeat_mean_rt_mae_rnn",
            "selected_repeat_mean_rt_mae_transformer",
            "selected_switch_mean_rt_mae_rnn",
            "selected_switch_mean_rt_mae_transformer",
        ]
        for label in ("G2G", "E2E", "E2G", "G2E"):
            paired_columns.extend(
                [
                    "selected_mean_rt_error_{}_rnn".format(label),
                    "selected_mean_rt_error_{}_transformer".format(label),
                ]
            )

        missing = [col for col in paired_columns if col not in self.paired.columns]
        if missing:
            raise KeyError(
                "Paired summary CSV missing required columns: {}".format(missing)
            )

        stat_columns = [
            "metric",
            "paired_t_p_two_sided",
            "holm_p_selected_condition_specific_RT",
            "holm_p_selected_repeat_switch_RT",
        ]
        missing = [
            col for col in stat_columns if col not in self.statistics_df.columns
        ]
        if missing:
            raise KeyError(
                "Statistics CSV missing required columns: {}".format(missing)
            )

        print("Data successfully loaded from {}".format(self.input_dir))

    def _plot_figure_get_stats(self):
        """Assemble the three architecture-ablation panels."""
        rc = {
            "svg.fonttype": "none",
            "path.simplify": False,
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
        }

        with plt.rc_context(rc):
            fig = plt.figure(figsize=self.figsize, dpi=self.figdpi)
            gs = fig.add_gridspec(
                1,
                3,
                width_ratios=[0.80, 1.45, 1.05],
                wspace=0.36,
            )

            ax_a = fig.add_subplot(gs[0, 0])
            ax_b = fig.add_subplot(gs[0, 1])
            ax_c = fig.add_subplot(gs[0, 2])

            self._make_panel_a(ax_a)
            self._make_panel_b(ax_b)
            self._make_panel_c(ax_c)

            for ax in (ax_a, ax_b, ax_c):
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(direction="out", length=3, width=0.8)

            fig.text(
                0.5,
                0.018,
                (
                    "Two-sided paired t-tests: "
                    "* p < 0.05, ** p < 0.01, *** p < 0.001, "
                    "**** p < 0.0001; n.s., not significant. "
                    "Condition-specific and repeat/switch p values are "
                    "Holm--Bonferroni adjusted within their respective families."
                ),
                ha="center",
                va="bottom",
                fontsize=8,
            )
            fig.subplots_adjust(
                left=0.075,
                right=0.99,
                top=0.92,
                bottom=0.20,
            )

        return fig

    def _make_panel_a(self, ax):
        """Panel a: selected-checkpoint overall RT MAE."""
        rnn, transformer = self._draw_paired_box(
            ax,
            self.paired["selected_mean_rt_mae_rnn"],
            self.paired["selected_mean_rt_mae_transformer"],
            x_rnn=0,
            x_transformer=1,
            edge_color=self.NEUTRAL_EDGE,
            transformer_fill=self.NEUTRAL_FILL,
            width=0.30,
        )

        p_value = self._get_stat_pvalue("selected_checkpoint_RT_MAE")
        ymin, ymax, yrange = self._value_range(rnn, transformer)
        bracket_y = ymax + 0.08 * yrange
        bracket_h = 0.035 * yrange

        self._add_significance_bracket(
            ax,
            0,
            1,
            bracket_y,
            bracket_h,
            self._p_to_stars(p_value),
        )

        ax.set_ylim(
            bottom=max(0, ymin - 0.10 * yrange),
            top=bracket_y + bracket_h + 0.14 * yrange,
        )
        ax.set_xlim(-0.38, 1.38)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["RNN", "Transformer"])
        ax.set_ylabel("RT MAE at selected checkpoint (ms)")
        ax.set_title("a  Overall RT fidelity", loc="left", fontweight="bold")

        print("Panel a p = {:.6g}".format(p_value))

    def _make_panel_b(self, ax):
        """Panel b: condition-specific selected-checkpoint RT errors."""
        labels = ["G2G", "E2E", "E2G", "G2E"]
        centers = np.arange(len(labels), dtype=float)
        offset = 0.18
        all_values = []
        bracket_tops = []

        for i, label in enumerate(labels):
            color = self.TRIAL_COLORS[label]
            x_rnn = centers[i] - offset
            x_tf = centers[i] + offset

            rnn, transformer = self._draw_paired_box(
                ax,
                self.paired["selected_mean_rt_error_{}_rnn".format(label)],
                self.paired[
                    "selected_mean_rt_error_{}_transformer".format(label)
                ],
                x_rnn=x_rnn,
                x_transformer=x_tf,
                edge_color=color,
                transformer_fill=color,
                width=0.24,
                point_size=14,
            )

            values = np.concatenate([rnn, transformer])
            all_values.append(values)

            local_min = np.nanmin(values)
            local_max = np.nanmax(values)
            local_range = max(local_max - local_min, 1.0)
            bracket_y = local_max + 0.10 * local_range
            bracket_h = 0.04 * local_range

            p_value = self._get_stat_pvalue(
                "selected_RT_error_{}".format(label),
                adjusted_column="holm_p_selected_condition_specific_RT",
            )
            self._add_significance_bracket(
                ax,
                x_rnn,
                x_tf,
                bracket_y,
                bracket_h,
                self._p_to_stars(p_value),
                fontsize=9.5,
            )
            bracket_tops.append(bracket_y + bracket_h)

            print(
                "Panel b {} adjusted p = {:.6g}".format(
                    label,
                    p_value,
                )
            )

        combined = np.concatenate(all_values)
        global_min = np.nanmin(combined)
        global_max = max(np.nanmax(combined), max(bracket_tops))
        global_range = max(global_max - global_min, 1.0)

        ax.set_ylim(
            bottom=max(0, global_min - 0.08 * global_range),
            top=global_max + 0.16 * global_range,
        )
        ax.set_xticks(centers)
        ax.set_xticklabels(labels)

        for tick, label in zip(ax.get_xticklabels(), labels):
            tick.set_color(self.TRIAL_COLORS[label])
            tick.set_fontweight("bold")

        ax.set_ylabel("Absolute RT error (ms)")
        ax.set_title(
            "b  Condition-specific RT error",
            loc="left",
            fontweight="bold",
        )
        ax.legend(
            handles=[
                Patch(
                    facecolor="white",
                    edgecolor=self.NEUTRAL_EDGE,
                    linewidth=1.1,
                    label="RNN (open)",
                ),
                Patch(
                    facecolor=self.NEUTRAL_FILL,
                    edgecolor=self.NEUTRAL_EDGE,
                    linewidth=1.1,
                    label="Transformer (filled)",
                ),
            ],
            frameon=False,
            loc="upper left",
        )

    def _make_panel_c(self, ax):
        """Panel c: repeat- versus switch-trial selected-checkpoint RT MAE."""
        group_labels = ["Repeat", "Switch"]
        group_metrics = [
            (
                "selected_repeat_mean_rt_mae_rnn",
                "selected_repeat_mean_rt_mae_transformer",
                "selected_repeat_RT_MAE",
            ),
            (
                "selected_switch_mean_rt_mae_rnn",
                "selected_switch_mean_rt_mae_transformer",
                "selected_switch_RT_MAE",
            ),
        ]
        centers = np.arange(2, dtype=float)
        offset = 0.18
        all_values = []
        bracket_tops = []

        for i, (rnn_key, tf_key, metric_name) in enumerate(group_metrics):
            x_rnn = centers[i] - offset
            x_tf = centers[i] + offset

            rnn, transformer = self._draw_paired_box(
                ax,
                self.paired[rnn_key],
                self.paired[tf_key],
                x_rnn=x_rnn,
                x_transformer=x_tf,
                edge_color=self.NEUTRAL_EDGE,
                transformer_fill=self.NEUTRAL_FILL,
                width=0.24,
                point_size=16,
            )

            values = np.concatenate([rnn, transformer])
            all_values.append(values)

            local_min = np.nanmin(values)
            local_max = np.nanmax(values)
            local_range = max(local_max - local_min, 1.0)
            bracket_y = local_max + 0.10 * local_range
            bracket_h = 0.04 * local_range

            p_value = self._get_stat_pvalue(
                metric_name,
                adjusted_column="holm_p_selected_repeat_switch_RT",
            )
            self._add_significance_bracket(
                ax,
                x_rnn,
                x_tf,
                bracket_y,
                bracket_h,
                self._p_to_stars(p_value),
            )
            bracket_tops.append(bracket_y + bracket_h)

            print(
                "Panel c {} adjusted p = {:.6g}".format(
                    group_labels[i],
                    p_value,
                )
            )

        combined = np.concatenate(all_values)
        global_min = np.nanmin(combined)
        global_max = max(np.nanmax(combined), max(bracket_tops))
        global_range = max(global_max - global_min, 1.0)

        ax.set_ylim(
            bottom=max(0, global_min - 0.08 * global_range),
            top=global_max + 0.16 * global_range,
        )
        ax.set_xticks(centers)
        ax.set_xticklabels(group_labels)
        ax.set_ylabel("RT MAE at selected checkpoint (ms)")
        ax.set_title(
            "c  Repeat- and switch-trial RT fidelity",
            loc="left",
            fontweight="bold",
        )

    def _get_stat_pvalue(self, metric, adjusted_column=None):
        """Return the p-value used for one figure annotation."""
        row = self.statistics_df.loc[self.statistics_df["metric"] == metric]
        if row.empty:
            return np.nan

        row = row.iloc[0]

        if (
                adjusted_column is not None
                and adjusted_column in row.index
                and np.isfinite(row[adjusted_column])
        ):
            return float(row[adjusted_column])

        if (
                "paired_t_p_two_sided" in row.index
                and np.isfinite(row["paired_t_p_two_sided"])
        ):
            return float(row["paired_t_p_two_sided"])

        return np.nan

    @staticmethod
    def _p_to_stars(p_value):
        """Convert a p-value to manuscript-style significance notation."""
        if p_value is None or not np.isfinite(p_value):
            return "n.s."
        if p_value < 1e-4:
            return "****"
        if p_value < 1e-3:
            return "***"
        if p_value < 1e-2:
            return "**"
        if p_value < 5e-2:
            return "*"
        return "n.s."

    @staticmethod
    def _value_range(*arrays):
        """Return minimum, maximum, and a non-zero plotting range."""
        values = np.concatenate(arrays)
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        return vmin, vmax, max(vmax - vmin, 1.0)

    @staticmethod
    def _add_significance_bracket(
            ax,
            x1,
            x2,
            y,
            h,
            text,
            fontsize=10,
    ):
        """Draw a significance bracket with stars or 'n.s.'."""
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + h, y + h, y],
            color="black",
            linewidth=0.9,
            clip_on=False,
            solid_capstyle="butt",
        )
        ax.text(
            (x1 + x2) / 2.0,
            y + h,
            text,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            clip_on=False,
        )

    @staticmethod
    def _style_boxplot(boxplot, edge_color, face_color, filled):
        """Apply manuscript styling to one boxplot."""
        for box in boxplot["boxes"]:
            box.set_edgecolor(edge_color)
            box.set_linewidth(1.15)
            box.set_facecolor(face_color if filled else "white")

        for median in boxplot["medians"]:
            median.set_color(edge_color)
            median.set_linewidth(1.35)

        for whisker in boxplot["whiskers"]:
            whisker.set_color(edge_color)
            whisker.set_linewidth(1.0)

        for cap in boxplot["caps"]:
            cap.set_color(edge_color)
            cap.set_linewidth(1.0)

    def _draw_paired_box(
            self,
            ax,
            rnn_values,
            transformer_values,
            x_rnn,
            x_transformer,
            edge_color,
            transformer_fill,
            width=0.25,
            point_size=18,
    ):
        """Draw paired subject points with RNN-open / Transformer-filled boxes."""
        rnn = np.asarray(rnn_values, dtype=float)
        transformer = np.asarray(transformer_values, dtype=float)
        valid = np.isfinite(rnn) & np.isfinite(transformer)
        rnn = rnn[valid]
        transformer = transformer[valid]

        for rnn_value, tf_value in zip(rnn, transformer):
            ax.plot(
                [x_rnn, x_transformer],
                [rnn_value, tf_value],
                color=self.PAIR_LINE_COLOR,
                linewidth=0.65,
                alpha=0.55,
                zorder=1,
            )

        rnn_box = ax.boxplot(
            [rnn],
            positions=[x_rnn],
            widths=width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            zorder=3,
        )
        self._style_boxplot(
            rnn_box,
            edge_color=edge_color,
            face_color="white",
            filled=False,
        )

        tf_box = ax.boxplot(
            [transformer],
            positions=[x_transformer],
            widths=width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            zorder=3,
        )
        self._style_boxplot(
            tf_box,
            edge_color=edge_color,
            face_color=transformer_fill,
            filled=True,
        )

        ax.scatter(
            np.full(len(rnn), x_rnn),
            rnn,
            s=point_size,
            facecolors="white",
            edgecolors=edge_color,
            linewidths=0.65,
            zorder=4,
        )
        ax.scatter(
            np.full(len(transformer), x_transformer),
            transformer,
            s=point_size,
            facecolors=transformer_fill,
            edgecolors=edge_color,
            linewidths=0.65,
            zorder=4,
        )

        return rnn, transformer
