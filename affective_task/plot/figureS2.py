"""Generate Extended Data Figure 2: reaction times across task conditions."""

import itertools
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from affective_task.utils import Constants, p_to_stars, save_figure


class FigureS2:
    """Compare participant RTs across pure, repeat, and switch conditions."""

    analysis_dir = "model_analysis"
    summary_fn = "summary.pkl"

    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "dataset"
    ats_data_fn = "ats_data_hddm_block_01.csv"

    figsize = (20, 10)
    figdpi = 300

    CONDITION_ORDER = [
        "Gender",
        "Emotion",
        "G2G",
        "E2E",
        "E2G",
        "G2E",
    ]

    CONDITION_COLORS = {
        "Gender": Constants.COLOR_REPEAT,
        "Emotion": Constants.COLOR_SWITCH,
        "G2G": Constants.COLOR_G2G,
        "E2E": Constants.COLOR_E2E,
        "E2G": Constants.COLOR_E2G,
        "G2E": Constants.COLOR_G2E,
    }

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

        self.pure_task_rt_df = None
        self.task_switch_rt_df = None

    def make_figure(self):
        """Generate, save, and return Extended Data Figure 2."""
        print("Making Figure S2...")
        self._run_preprocessing()

        print("Stats for Figure S2")
        print("-------------------")

        fig = self._plot_figure_get_stats()
        save_figure(
            fig,
            self.save_dir,
            "FigS2",
        )
        return fig

    def _run_preprocessing(self):
        """Prepare RT data for the six conditions shown in the figure."""
        self.task_switch_rt_df = self._load_task_switch_rt_data()
        self.pure_task_rt_df = self._load_pure_task_rt_data()

    def _load_task_switch_rt_data(self):
        """Load participant mean RTs for G2G, E2E, E2G, and G2E."""
        rows = []

        summary_keys = {
            "G2G": "u_mean_rt_trial_type_0",
            "E2E": "u_mean_rt_trial_type_1",
            "E2G": "u_mean_rt_trial_type_2",
            "G2E": "u_mean_rt_trial_type_3",
        }

        for patient_id in self.patients_id:
            summary_path = (
                    Path(self.model_dir)
                    / "s{}".format(patient_id)
                    / self.analysis_dir
                    / self.summary_fn
            )

            with summary_path.open("rb") as file:
                summary_stats = pickle.load(file)

            block_summary = summary_stats["01"]

            for transition, key in summary_keys.items():
                rows.append(
                    {
                        "subj_id": patient_id,
                        "transition": transition,
                        "rt": block_summary[key],
                    }
                )

        return pd.DataFrame(rows)

    def _load_pure_task_rt_data(self):
        """Load and filter RTs from the pure Gender and Emotion task blocks."""
        data_path = self.dataset_dir / self.ats_data_fn

        if not data_path.is_file():
            raise FileNotFoundError(
                "Missing required file: {}".format(data_path)
            )

        data = pd.read_csv(
            data_path,
            index_col=0,
        )

        # Remove RTs shorter than 200 ms.
        data = data.loc[
            data["rt"] >= 0.2
            ].copy()

        # Remove trials outside each participant's mean +/- 3 SD.
        subject_stats = (
            data
            .groupby("subj_idx")["rt"]
            .agg(
                mu="mean",
                sd=lambda values: values.std(ddof=1),
                n="count",
            )
            .reset_index()
        )

        data = data.merge(
            subject_stats,
            on="subj_idx",
            how="left",
        )

        data["lower"] = (
                data["mu"]
                - 3 * data["sd"]
        )
        data["upper"] = (
                data["mu"]
                + 3 * data["sd"]
        )

        no_dispersion = (
                data["sd"].isna()
                | (data["sd"] == 0)
        )
        within_bounds = (
                (data["rt"] >= data["lower"])
                & (data["rt"] <= data["upper"])
        )

        data = data.loc[
            no_dispersion | within_bounds
            ].drop(
            columns=[
                "mu",
                "sd",
                "n",
                "lower",
                "upper",
            ]
        )

        gender = data.loc[
            data["block"].isin([1, 2])
            & (data["task"] == "g"),
            [
                "subj_idx",
                "rt",
            ],
        ].copy()
        gender["transition"] = "Gender"

        emotion = data.loc[
            data["block"].isin([1, 2])
            & (data["task"] == "e"),
            [
                "subj_idx",
                "rt",
            ],
        ].copy()
        emotion["transition"] = "Emotion"

        pure_task_df = pd.concat(
            [
                gender,
                emotion,
            ],
            ignore_index=True,
        )

        # Convert seconds to milliseconds.
        pure_task_df["rt"] *= 1000.0
        pure_task_df = pure_task_df.rename(
            columns={
                "subj_idx": "subj_id",
            }
        )

        return pure_task_df[
            [
                "subj_id",
                "transition",
                "rt",
            ]
        ]

    def _plot_figure_get_stats(self):
        """Assemble the six-condition RT comparison."""
        fig = plt.figure(
            constrained_layout=False,
            figsize=self.figsize,
            dpi=self.figdpi,
        )
        gs = fig.add_gridspec(
            12,
            24,
        )

        all_trials_df = pd.concat(
            [
                self.pure_task_rt_df,
                self.task_switch_rt_df,
            ],
            ignore_index=True,
        )

        ax = fig.add_subplot(
            gs[7:12, 0:24]
        )
        self._make_panel_all_transitions(
            all_trials_df,
            ax,
        )

        return fig

    def _make_panel_all_transitions(
            self,
            df,
            ax,
    ):
        """Compare mean RT across all six pure/repeat/switch conditions."""
        order = self.CONDITION_ORDER
        xpos = {
            condition: index
            for index, condition in enumerate(order)
        }

        df = df.loc[
            df["transition"].isin(order)
        ].copy()

        # Subject-level mean RT for each condition.
        df_subj = (
            df
            .groupby(
                [
                    "subj_id",
                    "transition",
                ],
                as_index=False,
            )["rt"]
            .mean()
        )

        summary = (
            df_subj
            .groupby("transition")["rt"]
            .agg(
                mean_rt="mean",
                sd_rt="std",
                n="count",
            )
            .reset_index()
        )
        summary["sem_rt"] = (
                summary["sd_rt"]
                / np.sqrt(summary["n"])
        )
        summary["transition"] = pd.Categorical(
            summary["transition"],
            categories=order,
            ordered=True,
        )
        summary = (
            summary
            .sort_values("transition")
            .reset_index(drop=True)
        )

        print("\nMean and SEM for each trial type:")
        for _, row in summary.iterrows():
            print(
                "{}: mean = {:.3f} ms, SEM = {:.3f} ms, n = {}".format(
                    row["transition"],
                    row["mean_rt"],
                    row["sem_rt"],
                    int(row["n"]),
                )
            )

        palette = [
            self.CONDITION_COLORS[
                condition
            ]
            for condition in order
        ]

        sns.boxplot(
            data=df_subj,
            x="transition",
            y="rt",
            order=order,
            palette=palette,
            showcaps=True,
            fliersize=0,
            width=0.7,
            linewidth=0.5,
            ax=ax,
        )

        sns.stripplot(
            data=df_subj,
            x="transition",
            y="rt",
            order=order,
            palette=palette,
            alpha=0.8,
            size=4,
            jitter=False,
            zorder=10,
            linewidth=0.2,
            ax=ax,
        )

        # Connect each participant's available condition means.
        for _, group in df_subj.groupby(
                "subj_id",
                sort=False,
        ):
            group = group.set_index(
                "transition"
            )
            existing = [
                condition
                for condition in order
                if condition in group.index
            ]

            if len(existing) < 2:
                continue

            xs = [
                xpos[condition]
                for condition in existing
            ]
            ys = [
                group.loc[
                    condition,
                    "rt",
                ]
                for condition in existing
            ]

            ax.plot(
                xs,
                ys,
                color="gray",
                alpha=0.35,
                lw=0.8,
                zorder=2,
            )

        y_min_data = float(
            df_subj["rt"].min()
        )
        y_max_data = float(
            df_subj["rt"].max()
        )
        y_span = max(
            y_max_data - y_min_data,
            1.0,
        )

        # Keep the visible y-range close to the observed data.
        ax.set_ylim(
            y_min_data - 0.10 * y_span,
            y_max_data + 0.08 * y_span,
        )

        # One-sample t-tests against zero.
        one_sample_y = (
                y_max_data
                + 0.12 * y_span
        )

        for index, condition in enumerate(
                order
        ):
            values = df_subj.loc[
                df_subj["transition"] == condition,
                "rt",
            ].dropna()

            if len(values) >= 2:
                _, p_value = stats.ttest_1samp(
                    values,
                    0.0,
                )
                stars = p_to_stars(
                    p_value
                )
            else:
                stars = "n/a"

            ax.text(
                index,
                one_sample_y,
                stars,
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                clip_on=False,
                zorder=30,
            )

        pairwise_results = self._add_pairwise_comparisons(
            ax=ax,
            df_subj=df_subj,
            order=order,
            xpos=xpos,
            y_max_data=y_max_data,
            y_span=y_span,
        )

        print("\nPairwise paired t-test results:")
        print(
            pairwise_results
        )

        ax.set_xlabel(
            "Task transition"
        )
        ax.set_ylabel(
            "Mean RT (ms)"
        )

        # Make room above the axes for the significance annotations.
        plt.subplots_adjust(
            top=0.75
        )

    def _add_pairwise_comparisons(
            self,
            ax,
            df_subj,
            order,
            xpos,
            y_max_data,
            y_span,
    ):
        """Run and annotate all valid pairwise paired t-tests."""
        wide = df_subj.pivot(
            index="subj_id",
            columns="transition",
            values="rt",
        )

        comparisons = []

        for condition_1, condition_2 in itertools.combinations(
                order,
                2,
        ):
            pair_df = wide[
                [
                    condition_1,
                    condition_2,
                ]
            ].dropna()

            if len(pair_df) < 2:
                continue

            t_stat, p_value = stats.ttest_rel(
                pair_df[condition_1],
                pair_df[condition_2],
                nan_policy="omit",
            )

            comparisons.append(
                (
                    condition_1,
                    condition_2,
                    t_stat,
                    p_value,
                    len(pair_df),
                )
            )

        # Shorter brackets are drawn lower than longer brackets.
        comparisons = sorted(
            comparisons,
            key=lambda result: abs(
                xpos[result[1]]
                - xpos[result[0]]
            ),
        )

        base_y = (
                y_max_data
                + 0.18 * y_span
        )
        step_y = (
                0.08 * y_span
        )
        bracket_h = (
                0.02 * y_span
        )
        text_pad = (
                0.01 * y_span
        )

        results = []

        for index, (
                condition_1,
                condition_2,
                t_stat,
                p_value,
                n_used,
        ) in enumerate(comparisons):
            y = (
                    base_y
                    + index * step_y
            )

            self._add_sig_bracket_outside(
                ax=ax,
                x1=xpos[condition_1],
                x2=xpos[condition_2],
                y=y,
                h=bracket_h,
                text=p_to_stars(
                    p_value
                ),
                text_pad=text_pad,
                lw=1.0,
                fontsize=8,
            )

            results.append(
                {
                    "cond1": condition_1,
                    "cond2": condition_2,
                    "n": n_used,
                    "t": t_stat,
                    "p": p_value,
                }
            )

        return pd.DataFrame(
            results
        )

    @staticmethod
    def _add_sig_bracket_outside(
            ax,
            x1,
            x2,
            y,
            h,
            text,
            text_pad=2.0,
            lw=1.0,
            fontsize=8,
    ):
        """Draw a significance bracket outside the visible y-axis range."""
        ax.plot(
            [
                x1,
                x1,
                x2,
                x2,
            ],
            [
                y,
                y + h,
                y + h,
                y,
            ],
            lw=lw,
            color="black",
            clip_on=False,
            zorder=20,
        )
        ax.text(
            (x1 + x2) / 2.0,
            y + h + text_pad,
            text,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
            clip_on=False,
            zorder=21,
        )
