"""Generate Figure 6f-g: trial-bootstrap reliability comparisons."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from affective_task.utils import Constants


class Figure6FG:
    """Plot E2G/G2E trial-bootstrap split-half reliability distributions."""

    # Resolve the dataset directory from the GeoDynFormer project root.
    project_root = Path(__file__).resolve().parents[2]
    input_dir = project_root / "dataset"

    summary_csv = "trial_bootstrap_reliability_by_transition.csv"
    bootstrap_csv = "trial_bootstrap_distribution.csv"
    metadata_json = "trial_bootstrap_reliability_summary.json"

    figsize = (7.2, 3.4)
    figdpi = 300

    # Fixed manuscript axis limits used for both panels.
    xlim = (-0.3, 1.0)
    ylim = (0.4, 0.95)

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
        _ = rand_seed
        _ = n_boot

        self.summary_df = None
        self.boot_df = None
        self.metadata = None
        self.metric_name = None
        self.metric_label = None
        self.trial_filter = None

    def make_figure(self):
        """Generate, save, display, and return Figure 6f-g."""
        print("Making Figure 6fg...")
        self._run_preprocessing()

        print("Stats for Figure 6fg")
        print("--------------------")

        fig = self._plot_figure_get_stats()

        os.makedirs(
            self.save_dir,
            exist_ok=True,
        )

        png_path = os.path.join(
            self.save_dir,
            "Fig6fg.png",
        )
        svg_path = os.path.join(
            self.save_dir,
            "Fig6fg.svg",
        )

        fig.savefig(
            png_path,
            dpi=self.figdpi,
            bbox_inches="tight",
        )
        fig.savefig(
            svg_path,
            dpi=self.figdpi,
            bbox_inches="tight",
            format="svg",
        )

        print("Saved:")
        print("- {}".format(png_path))
        print("- {}".format(svg_path))

        plt.show()
        return fig

    def _run_preprocessing(self):
        """Load and validate the three precomputed trial-bootstrap files."""
        summary_path = os.path.join(
            self.input_dir,
            self.summary_csv,
        )
        bootstrap_path = os.path.join(
            self.input_dir,
            self.bootstrap_csv,
        )
        metadata_path = os.path.join(
            self.input_dir,
            self.metadata_json,
        )

        for path in (
                summary_path,
                bootstrap_path,
                metadata_path,
        ):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    "Missing required file: {}".format(path)
                )

        self.summary_df = pd.read_csv(
            summary_path
        )
        self.boot_df = pd.read_csv(
            bootstrap_path
        )

        with open(
                metadata_path,
                "r",
                encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

        required_summary = [
            "transition",
            "distance_metric",
            "distance_metric_label",
            "r_metric_sb_observed",
            "r_sc_sb_observed",
            "delta_fisher_z_observed",
            "ci95_delta_z_trial_boot_low",
            "ci95_delta_z_trial_boot_high",
            "p_delta_z_trial_bootstrap",
            "fraction_delta_z_gt_0",
        ]
        required_bootstrap = [
            "bootstrap",
            "transition",
            "r_metric_sb",
            "r_sc_sb",
            "delta_fisher_z",
        ]

        missing_summary = [
            column
            for column in required_summary
            if column not in self.summary_df.columns
        ]
        missing_bootstrap = [
            column
            for column in required_bootstrap
            if column not in self.boot_df.columns
        ]

        if missing_summary:
            raise KeyError(
                "Summary CSV missing required columns: {}".format(
                    missing_summary
                )
            )
        if missing_bootstrap:
            raise KeyError(
                "Bootstrap CSV missing required columns: {}".format(
                    missing_bootstrap
                )
            )

        if self.summary_df[
            "distance_metric"
        ].nunique() != 1:
            raise ValueError(
                "Summary CSV contains more than one distance metric."
            )

        self.metric_name = str(
            self.summary_df[
                "distance_metric"
            ].iloc[0]
        )
        self.metric_label = str(
            self.summary_df[
                "distance_metric_label"
            ].iloc[0]
        )

        self.trial_filter = str(
            self.metadata.get(
                "trial_filter",
                "unknown",
            )
        )
        if self.trial_filter not in {
            "correct",
            "all",
        }:
            self.trial_filter = "unknown"

        print(
            "Data successfully loaded from {}".format(
                self.input_dir
            )
        )
        print(
            "Distance metric: {}".format(
                self.metric_name
            )
        )
        print(
            "Trial filter: {}".format(
                self.trial_filter
            )
        )

    def _plot_figure_get_stats(self):
        """Assemble the E2G and G2E reliability panels."""
        fig, axes = plt.subplots(
            1,
            2,
            figsize=self.figsize,
            dpi=self.figdpi,
            constrained_layout=True,
        )

        transition_specs = [
            (
                "E2G",
                Constants.COLOR_E2G,
            ),
            (
                "G2E",
                Constants.COLOR_G2E,
            ),
        ]

        for ax, (
                transition,
                color,
        ) in zip(
            axes,
            transition_specs,
        ):
            summary = self._get_transition_summary(
                transition
            )
            boot_df = self.boot_df.loc[
                self.boot_df[
                    "transition"
                ] == transition
                ].copy()

            panel_stats = self._make_reliability_panel(
                ax=ax,
                boot_df=boot_df,
                summary=summary,
                transition_label=transition,
                color=color,
            )

            print(
                "{} bootstrap median: "
                "SC r_SB = {:.6f}; "
                "{} r_SB = {:.6f}".format(
                    transition,
                    panel_stats[
                        "median_sc"
                    ],
                    self.metric_label,
                    panel_stats[
                        "median_metric"
                    ],
                )
            )

        if self.trial_filter == "correct":
            title = "Correct trials"
        elif self.trial_filter == "all":
            title = "All trials"
        else:
            title = "Trial-bootstrap reliability"

        fig.suptitle(
            title,
            fontsize=11,
        )

        return fig

    def _get_transition_summary(
            self,
            transition,
    ):
        """Return the unique summary row for one transition."""
        subset = self.summary_df.loc[
            self.summary_df[
                "transition"
            ] == transition
            ].copy()

        if len(subset) != 1:
            raise ValueError(
                "Expected exactly one summary row for {}, found {}".format(
                    transition,
                    len(subset),
                )
            )

        return subset.iloc[
            0
        ].to_dict()

    def _make_reliability_panel(
            self,
            ax,
            boot_df,
            summary,
            transition_label,
            color,
    ):
        """Plot one transition's paired trial-bootstrap reliability cloud."""
        switch_cost_reliability = boot_df[
            "r_sc_sb"
        ].to_numpy(
            dtype=float
        )
        metric_reliability = boot_df[
            "r_metric_sb"
        ].to_numpy(
            dtype=float
        )
        delta_z = boot_df[
            "delta_fisher_z"
        ].to_numpy(
            dtype=float
        )

        valid = (
                np.isfinite(
                    switch_cost_reliability
                )
                & np.isfinite(
            metric_reliability
        )
                & np.isfinite(
            delta_z
        )
        )

        x = switch_cost_reliability[
            valid
        ]
        y = metric_reliability[
            valid
        ]
        delta_z = delta_z[
            valid
        ]

        if len(x) == 0:
            raise ValueError(
                "No valid bootstrap observations for {}.".format(
                    transition_label
                )
            )

        above = delta_z > 0
        below = delta_z < 0
        on_line = np.isclose(
            delta_z,
            0.0,
        )

        self._shade_above_identity(
            ax
        )

        if np.any(
                below
        ):
            ax.scatter(
                x[below],
                y[below],
                s=16,
                alpha=0.35,
                color="0.65",
                edgecolor="none",
                zorder=2,
                label=r"$\Delta z_b < 0$",
            )

        if np.any(
                above
        ):
            ax.scatter(
                x[above],
                y[above],
                s=16,
                alpha=0.35,
                color=color,
                edgecolor="none",
                zorder=3,
                label=r"$\Delta z_b > 0$",
            )

        if np.any(
                on_line
        ):
            ax.scatter(
                x[on_line],
                y[on_line],
                s=18,
                alpha=0.7,
                color="black",
                edgecolor="none",
                zorder=4,
                label=r"$\Delta z_b = 0$",
            )

        identity_low = max(
            self.xlim[0],
            self.ylim[0],
        )
        identity_high = min(
            self.xlim[1],
            self.ylim[1],
        )
        ax.plot(
            [
                identity_low,
                identity_high,
            ],
            [
                identity_low,
                identity_high,
            ],
            linestyle="--",
            color="black",
            linewidth=1.0,
            zorder=1,
        )

        # Coordinate-wise median of the bootstrap cloud.
        median_sc = float(
            np.median(x)
        )
        median_metric = float(
            np.median(y)
        )

        ax.scatter(
            median_sc,
            median_metric,
            s=52,
            marker="D",
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
            zorder=6,
            label="Bootstrap median",
        )

        ax.text(
            0.04,
            0.96,
            self._format_annotation(
                summary
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )

        ax.set_xlim(
            self.xlim
        )
        ax.set_ylim(
            self.ylim
        )

        ax.set_title(
            transition_label,
            fontsize=11,
            pad=6,
        )
        ax.set_xlabel(
            r"Switch cost reliability, "
            r"$r_{\mathrm{SB,SC},b}$"
        )
        ax.set_ylabel(
            "{}\n".format(
                self.metric_label
            )
            + r"reliability, "
            + r"$r_{\mathrm{SB,metric},b}$"
        )

        ax.legend(
            frameon=False,
            fontsize=8,
            loc="lower right",
        )

        return {
            "median_sc": median_sc,
            "median_metric": median_metric,
        }

    def _shade_above_identity(
            self,
            ax,
    ):
        """Shade the visible region in which metric reliability exceeds SC."""
        x_grid = np.linspace(
            self.xlim[0],
            self.xlim[1],
            300,
        )
        identity = x_grid
        upper = np.full_like(
            x_grid,
            self.ylim[1],
        )
        visible = (
                identity
                < self.ylim[1]
        )

        ax.fill_between(
            x_grid,
            np.maximum(
                identity,
                self.ylim[0],
            ),
            upper,
            where=visible,
            color="gray",
            alpha=0.08,
            zorder=0,
        )

    def _format_annotation(
            self,
            summary,
    ):
        """Return the manuscript annotation shown in each reliability panel."""
        fraction_above = (
                100.0
                * float(
            summary[
                "fraction_delta_z_gt_0"
            ]
        )
        )
        ci_low = float(
            summary[
                "ci95_delta_z_trial_boot_low"
            ]
        )
        ci_high = float(
            summary[
                "ci95_delta_z_trial_boot_high"
            ]
        )
        p_value = float(
            summary[
                "p_delta_z_trial_bootstrap"
            ]
        )

        annotation = (
            "$r_{{SB,metric}}$ = {:.3f}\n"
            "$r_{{SB,SC}}$ = {:.3f}\n"
            "$\\Delta z$ = {:.3f}\n"
            "{:.1f}% above diagonal\n"
            "95% CI$_{{\\Delta z}}$ = [{:.3f}, {:.3f}]\n"
        ).format(
            float(
                summary[
                    "r_metric_sb_observed"
                ]
            ),
            float(
                summary[
                    "r_sc_sb_observed"
                ]
            ),
            float(
                summary[
                    "delta_fisher_z_observed"
                ]
            ),
            fraction_above,
            ci_low,
            ci_high,
        )

        if p_value < 0.001:
            annotation += (
                "$p_{boot}$ <0.001"
            )
        else:
            annotation += (
                "$p_{{boot}}$ = {:.3f}".format(
                    p_value
                )
            )

        return annotation
