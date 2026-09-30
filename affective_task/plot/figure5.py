"""Generate Figure 5: latent trajectories, LDA, and fixed-point geometry."""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro, ttest_rel

from affective_task.utils import Constants, save_figure
from affective_task.visualization import PlotModelLatents


class Figure5:
    """Plot exemplar dynamics, LDA performance, and fixed-point distances."""

    analysis_dir = "model_analysis"
    stats_fn = "holdout_outputs_01SD.pkl"
    fp_summary_fn = "fixed_point_summary.pkl"
    lda_summary_fn = "lda_summary.pkl"
    patient_rs = "../../dataset/patient_rs.pkl"

    distance_keys = [
        "task_matched",
        "task_relevant",
        "task_irrelevant",
        "gender_relevant",
        "gender_irrelevant",
        "emotion_relevant",
        "emotion_irrelevant",
    ]

    example_user = "27"
    figsize = (7, 6.5)
    figdpi = 300

    def __init__(self, model_dir, save_dir, patients_id):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.patients_id = patients_id

        self.group_fp_summary = []
        self.group_lda_summary = []
        self.ex_stats = None

    def make_figure(self):
        """Generate and save Figure 5."""
        print("Making Figure 5...")
        self._run_preprocessing()
        print("Stats for Figure 5")
        print("------------------")

        fig = self._plot_figure_get_stats()
        save_figure(fig, self.save_dir, "Fig5")

    def _run_preprocessing(self):
        """Load model outputs, fixed-point summaries, and LDA summaries."""
        for patient_id in self.patients_id:
            subject_dir = os.path.join(self.model_dir, "s{}".format(patient_id))

            stats_path = os.path.join(
                subject_dir, self.analysis_dir, self.stats_fn
            )
            with open(stats_path, "rb") as path:
                expt_stats = pickle.load(path)

            if patient_id == self.example_user:
                self.ex_stats = expt_stats

            fp_summary_path = os.path.join(
                subject_dir, self.analysis_dir, self.fp_summary_fn
            )
            with open(fp_summary_path, "rb") as path:
                fp_summary = pickle.load(path)

            self.group_fp_summary.append(
                self._get_user_fp_stats(fp_summary, patient_id)
            )

            lda_summary_path = os.path.join(
                subject_dir, self.analysis_dir, self.lda_summary_fn
            )
            with open(lda_summary_path, "rb") as path:
                lda_summary = pickle.load(path)

            self.group_lda_summary.append(
                self._flatten_lda_summary(lda_summary, patient_id)
            )

        with open(self.patient_rs, "rb") as path:
            self.group_Rs = pickle.load(path)

        print("Data successfully loaded from {}".format(self.patient_rs))

    @staticmethod
    def _flatten_lda_summary(lda_summary, patient_id):
        """Flatten scalar and one-level nested LDA summary values."""
        flattened = {"expt": patient_id}

        for key, value in lda_summary.items():
            if np.isscalar(value):
                flattened[key] = value
            elif isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    flattened["{}_{}".format(key, nested_key)] = nested_value

        return pd.DataFrame(flattened, index=[0])

    def _plot_figure_get_stats(self):
        """Assemble Figure 5 and print the associated statistics."""
        fig = plt.figure(
            constrained_layout=False, figsize=self.figsize, dpi=self.figdpi
        )
        gs = fig.add_gridspec(11, 22)

        # Panel A: exemplar latent trajectories.
        ax_a = fig.add_subplot(gs[0:11, 0:11], projection="3d")
        plot_kwargs = {
            "xlim": [-6, 16],
            "ylim": [-9, 6],
            "zlim": [-20, 1],
            "colors": [
                          Constants.COLOR_G2G,
                          Constants.COLOR_E2E,
                          Constants.COLOR_E2G,
                          Constants.COLOR_G2E,
                      ] * 2,
            "line_styles": ["-"] * 4 + ["--"] * 4,
            "annotate": "global",
            "plot_t_posts": [1300] * 8,
            "R": self.group_Rs[self.example_user],
        }
        self._make_panel_A(ax_a, plot_kwargs)

        # Panels B1-B2: LDA misclassification distributions.
        ax_b1 = fig.add_subplot(gs[0:2, 13:20])
        self._make_panel_B1(ax_b1)

        ax_b2 = fig.add_subplot(gs[3:5, 13:20])
        self._make_panel_B2(ax_b2)

        group_lda_df = pd.concat(self.group_lda_summary, ignore_index=True)
        self._print_all_lda_group_stats(group_lda_df)

        # Panels C1-C2: fixed-point distance hierarchy.
        group_fp_df = self._get_group_fp_stats()

        ax_c1 = fig.add_subplot(gs[6:8, 13:20])
        self._make_panel_C1(ax_c1, group_fp_df)

        ax_c2 = fig.add_subplot(gs[9:11, 13:20])
        self._make_panel_C2(ax_c2, group_fp_df)

        return fig

    def _make_panel_A(self, ax, plot_kwargs, elev=40, azim=60, t_post=1400):
        """Plot all exemplar trajectories in the aligned UMAP space."""
        plotter = PlotModelLatents(
            self.ex_stats, post_on_dur=t_post, plot_pre_onset=False
        )
        plotter.plot_full_conditions(
            ax,
            elev=elev,
            azim=azim,
            plot_task_centroid=False,
            **plot_kwargs
        )

    def _make_panel_B1(self, ax):
        """Plot between-task and within-task LDA errors."""
        self._plot_lda_kde(
            ax=ax,
            keys=[
                "bw_error",
                "bw_shuffle_error",
                "within_error",
                "within_shuffle_error",
            ],
            labels={
                "bw_error": "Between task",
                "bw_shuffle_error": "Between task shuffle",
                "within_error": "Within task",
                "within_shuffle_error": "Within task shuffle",
            },
            palette=[
                Constants.COLOR_G2G,
                Constants.COLOR_E2E,
                Constants.COLOR_E2G,
                Constants.COLOR_G2E,
            ],
        )

    def _make_panel_B2(self, ax):
        """Plot gender and emotion LDA errors."""
        self._plot_lda_kde(
            ax=ax,
            keys=[
                "gender_error",
                "gender_shuffle_error",
                "emotion_error",
                "emotion_shuffle_error",
            ],
            labels={
                "gender_error": "Gender",
                "gender_shuffle_error": "Gender shuffle",
                "emotion_error": "Emotion",
                "emotion_shuffle_error": "Emotion shuffle",
            },
            palette=[
                Constants.COLOR_G2G,
                Constants.COLOR_REPEAT,
                Constants.COLOR_E2E,
                Constants.COLOR_SWITCH,
            ],
        )

    def _plot_lda_kde(self, ax, keys, labels, palette):
        """Plot subject-level observed and shuffle-control LDA errors."""
        df = pd.concat(self.group_lda_summary, ignore_index=True)
        df_plot = pd.melt(df, id_vars=["expt"], value_vars=keys).replace(labels)

        sns.kdeplot(
            data=df_plot,
            x="value",
            hue="variable",
            palette=palette,
            ax=ax,
            common_norm=False,
            cumulative=False,
            linewidth=1,
        )

        # Balanced binary classification has chance error = 0.5.
        ax.axvline(0.5, color="0.5", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Misclassification rate")
        ax.set_ylabel("Density")

        legend = ax.get_legend()
        legend.set_title(None)
        for text in legend.get_texts():
            text.set_fontsize(4)
        legend.get_frame().set_linewidth(0.0)

    def _get_lda_group_test(self, df, prefix):
        """Run one paired subject-level observed-vs-shuffle LDA test."""
        observed = df["{}_error".format(prefix)].to_numpy(dtype=float)
        shuffle = df["{}_shuffle_error".format(prefix)].to_numpy(dtype=float)

        valid = np.isfinite(observed) & np.isfinite(shuffle)
        observed = observed[valid]
        shuffle = shuffle[valid]

        if len(observed) < 2:
            return {
                "prefix": prefix,
                "n": len(observed),
                "obs": observed,
                "shuffle": shuffle,
                "t": np.nan,
                "p": np.nan,
                "diff": np.array([], dtype=float),
                "shapiro_w": np.nan,
                "shapiro_p": np.nan,
                "complete_nonoverlap": False,
                "empirical_gap": np.nan,
            }

        t_stat, p_value = ttest_rel(observed, shuffle)
        differences = observed - shuffle

        # Normality for a paired t-test concerns the paired differences.
        if 3 <= len(differences) <= 5000 and np.std(differences, ddof=1) > 0:
            shapiro_w, shapiro_p = shapiro(differences)
        else:
            shapiro_w, shapiro_p = np.nan, np.nan

        observed_max = np.max(observed)
        shuffle_min = np.min(shuffle)
        complete_nonoverlap = observed_max < shuffle_min
        empirical_gap = shuffle_min - observed_max if complete_nonoverlap else 0.0

        return {
            "prefix": prefix,
            "n": len(observed),
            "obs": observed,
            "shuffle": shuffle,
            "t": t_stat,
            "p": p_value,
            "diff": differences,
            "shapiro_w": shapiro_w,
            "shapiro_p": shapiro_p,
            "complete_nonoverlap": complete_nonoverlap,
            "empirical_gap": empirical_gap,
        }

    def _print_lda_shuffle_stats(self, df, prefix, adjusted_p=None):
        """Print descriptive and inferential statistics for one LDA analysis."""
        result = self._get_lda_group_test(df, prefix)
        observed = result["obs"]
        shuffle = result["shuffle"]
        n = result["n"]

        print("-" * 60)
        print(prefix.upper())

        if n == 0:
            print("No valid paired observations.")
            return result

        observed_mean = np.mean(observed)
        observed_sem = np.std(observed, ddof=1) / np.sqrt(n) if n > 1 else np.nan
        shuffle_mean = np.mean(shuffle)
        shuffle_sem = np.std(shuffle, ddof=1) / np.sqrt(n) if n > 1 else np.nan

        print(
            "Observed CV error: {:.4f} +/- {:.4f}".format(
                observed_mean, observed_sem
            )
        )
        print(
            "Mean shuffle-control CV error: {:.4f} +/- {:.4f}".format(
                shuffle_mean, shuffle_sem
            )
        )
        print(
            "Deviation of shuffle-control mean from chance (0.5): {:+.4f}".format(
                shuffle_mean - 0.5
            )
        )
        print(
            "Observed empirical range: [{:.4f}, {:.4f}]".format(
                np.min(observed), np.max(observed)
            )
        )
        print(
            "Subject-level mean shuffle-control empirical range: "
            "[{:.4f}, {:.4f}]".format(np.min(shuffle), np.max(shuffle))
        )
        print(
            "Complete empirical non-overlap "
            "(max observed < min mean-shuffle): {}".format(
                result["complete_nonoverlap"]
            )
        )
        print("Empirical range gap: {:.4f}".format(result["empirical_gap"]))
        print(
            "Paired t-test, observed CV error vs. subject-level mean "
            "shuffle-control CV error: t({}) = {:.6f}, raw p = {:.6g}".format(
                n - 1, result["t"], result["p"]
            )
        )

        if adjusted_p is not None:
            print(
                "Holm-Bonferroni-adjusted p across the four LDA "
                "group-level tests: {:.6g}".format(adjusted_p)
            )

        print(
            "Shapiro-Wilk test of paired differences "
            "(diagnostic for paired t-test): W = {:.6f}, p = {:.6g}".format(
                result["shapiro_w"], result["shapiro_p"]
            )
        )
        return result

    def _print_all_lda_group_stats(self, df):
        """Run and report the four planned group-level LDA tests."""
        prefixes = ["bw", "within", "gender", "emotion"]
        results = {
            prefix: self._get_lda_group_test(df, prefix)
            for prefix in prefixes
        }

        raw_pvalues = [results[prefix]["p"] for prefix in prefixes]
        if np.all(np.isfinite(raw_pvalues)):
            adjusted_pvalues = self._holm_bonferroni(raw_pvalues)
        else:
            adjusted_pvalues = np.full(len(prefixes), np.nan)

        print("LDA cross-validated observed vs. shuffle-control statistics:")
        print(
            "Group inference: paired t-tests across subjects, comparing "
            "observed CV error with each subject's mean CV error across "
            "100 label shuffles."
        )
        print(
            "Holm-Bonferroni correction is reported across the four "
            "group-level LDA tests (bw, within, gender, emotion)."
        )

        for prefix, adjusted_p in zip(prefixes, adjusted_pvalues):
            self._print_lda_shuffle_stats(
                df, prefix, adjusted_p=adjusted_p
            )

        print("---------------------------------------")
        return results

    @staticmethod
    def _holm_bonferroni(p_values):
        """Return Holm-Bonferroni-adjusted p-values in the original order."""
        p_values = np.asarray(p_values, dtype=float)
        n_tests = len(p_values)
        order = np.argsort(p_values)
        adjusted_sorted = np.empty(n_tests, dtype=float)

        running_max = 0.0
        for rank, index in enumerate(order):
            adjusted = (n_tests - rank) * p_values[index]
            running_max = max(running_max, adjusted)
            adjusted_sorted[rank] = min(running_max, 1.0)

        adjusted = np.empty(n_tests, dtype=float)
        for rank, index in enumerate(order):
            adjusted[index] = adjusted_sorted[rank]

        return adjusted

    def _make_panel_C1(self, ax, df):
        """Plot the controlled three-level fixed-point hierarchy."""
        keys = ["task_matched", "task_relevant", "task_irrelevant"]
        self._plot_fixed_point_kde(
            ax=ax,
            df=df,
            keys=keys,
            labels={
                "task_matched": "Between task\n(attributes matched)",
                "task_relevant": "Within task\ntask-relevant contrast",
                "task_irrelevant": "Within task\ntask-irrelevant contrast",
            },
            palette=[
                Constants.COLOR_E2G,
                Constants.COLOR_G2G,
                Constants.COLOR_E2E,
            ],
        )

        print("Stats on controlled fixed-point hierarchy:")
        self._print_distance_descriptives(df, keys)

        t_task_vs_rel, p_task_vs_rel = ttest_rel(
            df["task_matched"].values,
            df["task_relevant"].values,
            nan_policy="omit",
        )
        t_rel_vs_irr, p_rel_vs_irr = ttest_rel(
            df["task_relevant"].values,
            df["task_irrelevant"].values,
            nan_policy="omit",
        )
        adjusted_p = self._holm_bonferroni(
            [p_task_vs_rel, p_rel_vs_irr]
        )

        print(
            "Controlled hierarchy, task vs. task-relevant contrast: "
            "t = {}, raw p = {}, Holm-Bonferroni-adjusted p = {}, N = {}".format(
                t_task_vs_rel, p_task_vs_rel, adjusted_p[0], len(df)
            )
        )
        print(
            "Controlled hierarchy, task-relevant vs. task-irrelevant contrast: "
            "t = {}, raw p = {}, Holm-Bonferroni-adjusted p = {}, N = {}".format(
                t_rel_vs_irr, p_rel_vs_irr, adjusted_p[1], len(df)
            )
        )

    def _make_panel_C2(self, ax, df):
        """Plot task-specific relevant vs. irrelevant fixed-point contrasts."""
        keys = [
            "gender_relevant",
            "gender_irrelevant",
            "emotion_relevant",
            "emotion_irrelevant",
        ]
        self._plot_fixed_point_kde(
            ax=ax,
            df=df,
            keys=keys,
            labels={
                "gender_relevant": "Within Gender,\ntask-relevant contrast",
                "gender_irrelevant": "Within Gender,\ntask-irrelevant contrast",
                "emotion_relevant": "Within Emotion,\ntask-relevant contrast",
                "emotion_irrelevant": "Within Emotion,\ntask-irrelevant contrast",
            },
            palette=[
                Constants.COLOR_G2G,
                Constants.COLOR_REPEAT,
                Constants.COLOR_E2E,
                Constants.COLOR_SWITCH,
            ],
        )

        print("Task-specific controlled fixed-point contrasts:")
        self._print_distance_descriptives(df, keys)

        t_gender, p_gender = ttest_rel(
            df["gender_relevant"].values,
            df["gender_irrelevant"].values,
            nan_policy="omit",
        )
        t_emotion, p_emotion = ttest_rel(
            df["emotion_relevant"].values,
            df["emotion_irrelevant"].values,
            nan_policy="omit",
        )
        adjusted_p = self._holm_bonferroni([p_gender, p_emotion])

        print(
            "Gender task, relevant vs. irrelevant contrast: "
            "t = {}, raw p = {}, Holm-Bonferroni-adjusted p = {}, N = {}".format(
                t_gender, p_gender, adjusted_p[0], len(df)
            )
        )
        print(
            "Emotion task, relevant vs. irrelevant contrast: "
            "t = {}, raw p = {}, Holm-Bonferroni-adjusted p = {}, N = {}".format(
                t_emotion, p_emotion, adjusted_p[1], len(df)
            )
        )

    @staticmethod
    def _plot_fixed_point_kde(ax, df, keys, labels, palette):
        """Plot KDEs of fixed-point Euclidean-distance contrasts."""
        df_plot = pd.melt(
            df, id_vars=["expt"], value_vars=keys
        ).replace(labels)

        sns.kdeplot(
            data=df_plot,
            x="value",
            hue="variable",
            palette=palette,
            ax=ax,
            common_norm=False,
            cumulative=False,
            linewidth=1,
        )

        ax.set_xlabel("Euclidean distance\nbetween fixed points (a.u.)")
        ax.set_ylabel("Density")

        legend = ax.get_legend()
        legend.set_title(None)
        for text in legend.get_texts():
            text.set_fontsize(4)
        legend.get_frame().set_linewidth(0.0)

    @staticmethod
    def _print_distance_descriptives(df, keys):
        """Print mean ± SEM Euclidean distance for each contrast."""
        for key in keys:
            values = df[key].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            mean_value = np.mean(values)
            sem_value = np.std(values, ddof=1) / np.sqrt(len(values))

            print(
                "{} mean +/- s.e.m. Euclidean distance: "
                "{} +/- {}, N = {}".format(
                    key, mean_value, sem_value, len(values)
                )
            )

    def _get_user_fp_stats(self, data, experiment_id):
        """Summarize one participant's fixed-point distance statistics."""
        summary = {}
        for key in self.distance_keys:
            summary[key] = (
                np.nan if len(data[key]) == 0 else np.mean(data[key])
            )

        summary["expt"] = experiment_id
        summary["N"] = data["N"]
        summary["f_stimuli_with_fp"] = data["f_stimuli_with_fp"]
        return pd.DataFrame(summary, index=[0])

    def _get_group_fp_stats(self):
        """Combine fixed-point summaries and report inclusion statistics."""
        print("Stats on number of fixed points, all models:")

        df = pd.concat(self.group_fp_summary, ignore_index=True)
        n_zero = len(df.query("N == 0"))
        print("N models with no fixed points: {}".format(n_zero))
        print("--------------------------------------------")

        for key in self.distance_keys:
            print(
                "N models with no pairs for {}: {}".format(
                    key, df[key].isna().sum()
                )
            )

        # Keep the original complete-case criterion across all distance
        # contrasts so panels C1 and C2 use the same participant set.
        df_filtered = df.dropna(axis=0, how="any")

        n_at_least_ten = len(df_filtered.query("N >= 10"))
        n_mean = df_filtered["N"].mean()
        n_sem = df_filtered["N"].sem()
        fraction_mean = df_filtered["f_stimuli_with_fp"].mean()
        fraction_sem = df_filtered["f_stimuli_with_fp"].sem()

        print("Stats on fixed points, models included in distance analyses:")
        print(
            "N models included in distance analyses: {}".format(
                len(df_filtered)
            )
        )
        print(
            "N models with at least ten fixed points: {}".format(
                n_at_least_ten
            )
        )
        print(
            "Mean +/- s.e.m. fixed points per model: {} +/- {}".format(
                n_mean, n_sem
            )
        )
        print(
            "Mean +/- s.e.m. fraction of possible stimulus configurations "
            "with a fixed point: {} +/- {}".format(
                fraction_mean, fraction_sem
            )
        )
        print("------------------------------------------------------------")
        return df_filtered
