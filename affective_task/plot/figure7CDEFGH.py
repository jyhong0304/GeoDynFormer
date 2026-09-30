"""Generate Figure 7c-h: trajectory geometry and HDDM relationships."""

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import (
    PearsonRConstantInputWarning,
    pearsonr,
    sem,
    ttest_rel,
)
from statsmodels.stats.multitest import multipletests

from affective_task.utils import Constants, p_to_stars, save_figure


class Figure7CDEFGH:
    """Plot trajectory geometry/HDDM relationships for E2G and G2E trials.

    The active analysis uses the final manuscript configuration:

    - native latent space (all dimensions);
    - ``tc_mode="closest"``;
    - ``tc_reference="target_repeat_onset"``;
    - equal weighting across the four stimulus conditions.

    ``bar_F_0_tc`` and ``bar_F_tc_rt`` are retained as column names for
    compatibility with the existing plotting code, but they contain
    tortuosity values T = L / D.
    """

    analysis_dir = "model_analysis"
    stats_fn = "holdout_outputs_01SD.pkl"

    dataset_path = "../../dataset/"
    hddm_stats_fn = "HDDM_group_results_vat_all-trials.csv"

    figsize = (14, 10)
    figdpi = 300

    def __init__(self, model_dir, save_dir, patients_id, rand_seed, n_boot):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.patients_id = patients_id

        # Retain the common figure-class constructor signature.
        _ = rand_seed
        _ = n_boot

        warnings.simplefilter(
            "ignore",
            PearsonRConstantInputWarning,
        )

        self.hddm_stats = pd.read_csv(
            os.path.join(
                self.dataset_path,
                self.hddm_stats_fn,
            ),
            index_col=0,
        )

        self.subj_exps = {}
        self.hddm_by_subject = {}

    def make_figure(self):
        """Generate and save Figure 7c-h."""
        print("Making Figure 7cdefgh...")
        self._run_preprocessing()

        print("Stats for Figure 7cdefgh")
        print("------------------------")

        fig = self._plot_figure_get_stats()
        save_figure(
            fig,
            self.save_dir,
            "Fig7cdefgh",
        )
        return fig

    def _run_preprocessing(self):
        """Load participant model outputs and HDDM parameters."""
        for patient_id in self.patients_id:
            stats_path = os.path.join(
                self.model_dir,
                "s{}".format(patient_id),
                self.analysis_dir,
                self.stats_fn,
            )
            with open(stats_path, "rb") as path:
                self.subj_exps[patient_id] = pickle.load(path)

            drift_rate_df = self.hddm_stats.loc[
                (self.hddm_stats["subj_id"] == str(patient_id))
                & (self.hddm_stats["param"] == "drift rate")
                ]
            nondecision_df = self.hddm_stats.loc[
                (self.hddm_stats["subj_id"] == str(patient_id))
                & (self.hddm_stats["param"] == "non decision time")
                ]

            self.hddm_by_subject[patient_id] = {
                "E2G": {
                    "drift_rate": drift_rate_df.loc[
                        drift_rate_df["trial"] == "Gender Switch Trials",
                        "mean",
                    ].iloc[0],
                    "nondecision_time": nondecision_df.loc[
                        nondecision_df["trial"] == "Gender Switch Trials",
                        "mean",
                    ].iloc[0],
                },
                "G2E": {
                    "drift_rate": drift_rate_df.loc[
                        drift_rate_df["trial"] == "Emotion Switch Trials",
                        "mean",
                    ].iloc[0],
                    "nondecision_time": nondecision_df.loc[
                        nondecision_df["trial"] == "Emotion Switch Trials",
                        "mean",
                    ].iloc[0],
                },
            }

    def _plot_figure_get_stats(self):
        """Assemble Figure 7c-h and print the associated statistics."""
        fig = plt.figure(
            constrained_layout=False,
            figsize=self.figsize,
            dpi=self.figdpi,
        )
        gs = fig.add_gridspec(10, 16)

        df_geom_hddm = self._compute_group_geometry_hddm_dataframe()

        # ------------------------------------------------------------------
        # Figure 7c-f: HDDM parameters vs. epoch-specific tortuosity.
        # ------------------------------------------------------------------
        ax_c_e2g = fig.add_subplot(gs[0:4, 0:4])
        ax_c_g2e = fig.add_subplot(gs[0:4, 6:10])
        ax_d_e2g = fig.add_subplot(gs[6:10, 0:4])
        ax_d_g2e = fig.add_subplot(gs[6:10, 6:10])

        stats_df = self._plot_tc_hddm_four_panels(
            axes=[
                ax_c_e2g,
                ax_c_g2e,
                ax_d_e2g,
                ax_d_g2e,
            ],
            df_geom_hddm=df_geom_hddm,
            plot_boot_band=True,
            n_boot_band=10000,
            ci=95,
            seed=0,
        )

        with pd.option_context(
                "display.max_columns",
                None,
        ):
            print(stats_df)

        self._print_holm_corrections(stats_df)

        # ------------------------------------------------------------------
        # Figure 7g-h: E2G vs. G2E early geometry.
        # ------------------------------------------------------------------
        ax_early_length = fig.add_subplot(
            gs[0:4, 12:16]
        )
        self._make_transition_metric_boxplot(
            df=df_geom_hddm,
            ax=ax_early_length,
            metric="bar_L_0_tc",
            metric_label=r"Early trajectory length $\bar{L}_{[0,t_c]}$",
        )

        ax_early_tortuosity = fig.add_subplot(
            gs[6:10, 12:16]
        )
        self._make_transition_metric_boxplot(
            df=df_geom_hddm,
            ax=ax_early_tortuosity,
            metric="bar_F_0_tc",
            metric_label=r"Early trajectory tortuosity $\bar{T}_{[0,t_c]}$",
        )

        return fig

    @staticmethod
    def _switch_spec(switch_type):
        """Return switch and target-repeat task cues for one transition."""
        switch_type = switch_type.upper()

        if switch_type == "E2G":
            return {
                "switch_task_cue": 0,
                "switch_prev_task_cue": 1,
                "reference_task_cue": 0,
                "reference_prev_task_cue": 0,
            }

        if switch_type == "G2E":
            return {
                "switch_task_cue": 1,
                "switch_prev_task_cue": 0,
                "reference_task_cue": 1,
                "reference_prev_task_cue": 1,
            }

        raise ValueError(
            "switch_type must be either 'E2G' or 'G2E'."
        )

    @staticmethod
    def _path_length(points):
        """Return the cumulative Euclidean path length of a trajectory."""
        points = np.asarray(
            points,
            dtype=float,
        )

        if points.ndim != 2:
            raise ValueError(
                "points must have shape (T, D)."
            )
        if points.shape[0] < 2:
            return np.nan

        diffs = np.diff(
            points,
            axis=0,
        )
        return float(
            np.nansum(
                np.linalg.norm(
                    diffs,
                    axis=1,
                )
            )
        )

    @staticmethod
    def _mean_rt_index(exp, trial_pos, rt_col="mrt_ms"):
        """Convert the mean model RT of selected trials to a trajectory index."""
        if rt_col not in exp.df.columns:
            raise ValueError(
                "{} is not in exp.df.".format(rt_col)
            )

        mean_rt_ms = float(
            np.nanmean(
                exp.df.iloc[trial_pos][rt_col].to_numpy()
            )
        )
        rt_steps = int(
            np.round(
                mean_rt_ms / exp.step
            )
        )
        rt_index = int(
            exp.n_pre + rt_steps
        )

        return rt_index, mean_rt_ms

    @staticmethod
    def _compute_condition_mean_trajectory(
            exp,
            task_cue,
            prev_task_cue,
            stim_gender,
            stim_emotion,
            latent_key="latents",
            correct_only=True,
            model_correct_col="mcorrect",
            prev_model_correct_col="m_prev_correct",
    ):
        """Return the mean latent trajectory for one task/stimulus condition."""
        df = exp.df

        mask = (
                (df["task_cue"] == task_cue)
                & (df["prev_task_cue"] == prev_task_cue)
                & (df["stim_gender"] == stim_gender)
                & (df["stim_emotion"] == stim_emotion)
        )

        if correct_only:
            if model_correct_col in df.columns:
                mask = mask & (
                        df[model_correct_col] == 1
                )
            if prev_model_correct_col in df.columns:
                mask = mask & (
                        df[prev_model_correct_col] == 1
                )

        trial_pos = np.flatnonzero(
            mask.to_numpy()
        )
        if len(trial_pos) == 0:
            raise ValueError(
                "No trials found for task_cue={}, prev_task_cue={}, "
                "stim_gender={}, stim_emotion={}.".format(
                    task_cue,
                    prev_task_cue,
                    stim_gender,
                    stim_emotion,
                )
            )

        if latent_key not in exp.windowed:
            raise KeyError(
                "{} is not found in exp.windowed.".format(
                    latent_key
                )
            )

        latents = exp.windowed[latent_key]
        selected_latents = latents[
                           :,
                           trial_pos,
                           :,
                           ]
        mean_trajectory = np.nanmean(
            selected_latents,
            axis=1,
        )

        return mean_trajectory, trial_pos

    def _compute_condition_geometry(
            self,
            exp,
            switch_type,
            stim_gender,
            stim_emotion,
            latent_key="latents",
            rt_col="mrt_ms",
            correct_only=True,
    ):
        """Compute final-config t_c and trajectory metrics for one condition.

        ``t_c`` is the point on the switch trajectory that is closest in
        Euclidean distance to the onset state of the corresponding target-task
        repeat trajectory.
        """
        spec = self._switch_spec(
            switch_type
        )

        switch_trajectory, switch_pos = (
            self._compute_condition_mean_trajectory(
                exp=exp,
                task_cue=spec["switch_task_cue"],
                prev_task_cue=spec["switch_prev_task_cue"],
                stim_gender=stim_gender,
                stim_emotion=stim_emotion,
                latent_key=latent_key,
                correct_only=correct_only,
            )
        )

        reference_trajectory, reference_pos = (
            self._compute_condition_mean_trajectory(
                exp=exp,
                task_cue=spec["reference_task_cue"],
                prev_task_cue=spec["reference_prev_task_cue"],
                stim_gender=stim_gender,
                stim_emotion=stim_emotion,
                latent_key=latent_key,
                correct_only=correct_only,
            )
        )

        onset_idx = 0
        reference_onset = reference_trajectory[
            onset_idx
        ]

        rt_index, mean_rt_ms = self._mean_rt_index(
            exp=exp,
            trial_pos=switch_pos,
            rt_col=rt_col,
        )
        rt_index = int(
            np.clip(
                rt_index,
                onset_idx + 1,
                switch_trajectory.shape[0] - 1,
            )
        )

        search_trajectory = switch_trajectory[
                            onset_idx: rt_index + 1
                            ]
        if search_trajectory.shape[0] < 2:
            raise ValueError(
                "Search trajectory is too short to compute t_c."
            )

        distance_to_reference = np.linalg.norm(
            search_trajectory
            - reference_onset[None, :],
            axis=1,
        )
        tc_relative = int(
            np.nanargmin(
                distance_to_reference
            )
        )
        tc_index = onset_idx + tc_relative

        trajectory_0_tc = switch_trajectory[
                          onset_idx: tc_index + 1
                          ]
        trajectory_tc_rt = switch_trajectory[
                           tc_index: rt_index + 1
                           ]
        trajectory_0_rt = switch_trajectory[
                          onset_idx: rt_index + 1
                          ]

        length_0_tc = self._path_length(
            trajectory_0_tc
        )
        length_tc_rt = self._path_length(
            trajectory_tc_rt
        )
        length_0_rt = self._path_length(
            trajectory_0_rt
        )

        z_onset = switch_trajectory[onset_idx]
        z_tc = switch_trajectory[tc_index]
        z_rt = switch_trajectory[rt_index]

        displacement_0_tc = float(
            np.linalg.norm(
                z_tc - z_onset
            )
        )
        displacement_tc_rt = float(
            np.linalg.norm(
                z_rt - z_tc
            )
        )
        displacement_0_rt = float(
            np.linalg.norm(
                z_rt - z_onset
            )
        )

        tc_ms = float(
            (tc_index - onset_idx) * exp.step
        )
        rt_minus_tc_ms = float(
            mean_rt_ms - tc_ms
        )

        tortuosity_0_tc = self._tortuosity(
            length_0_tc,
            displacement_0_tc,
            tc_ms,
        )
        tortuosity_tc_rt = self._tortuosity(
            length_tc_rt,
            displacement_tc_rt,
            rt_minus_tc_ms,
        )
        tortuosity_0_rt = self._tortuosity(
            length_0_rt,
            displacement_0_rt,
            mean_rt_ms,
        )

        return {
            "stim_gender": stim_gender,
            "stim_emotion": stim_emotion,
            "n_switch": int(
                len(switch_pos)
            ),
            "n_reference": int(
                len(reference_pos)
            ),
            "tc_ms": tc_ms,
            "mean_rt_ms_switch": mean_rt_ms,
            "rt_minus_tc_ms": rt_minus_tc_ms,
            "L_0_tc": length_0_tc,
            "L_tc_rt": length_tc_rt,
            "L_0_rt": length_0_rt,
            "D_0_tc": displacement_0_tc,
            "D_tc_rt": displacement_tc_rt,
            "D_0_rt": displacement_0_rt,
            "F_0_tc": tortuosity_0_tc,
            "F_tc_rt": tortuosity_tc_rt,
            "F_0_rt": tortuosity_0_rt,
            "min_dist_to_reference": float(
                np.nanmin(
                    distance_to_reference
                )
            ),
        }

    @staticmethod
    def _tortuosity(path_length, displacement, duration_ms):
        """Return T = L / D for a valid trajectory epoch."""
        if (
                not np.isfinite(path_length)
                or not np.isfinite(displacement)
                or not np.isfinite(duration_ms)
        ):
            return np.nan

        if displacement <= 0 or duration_ms <= 0:
            return np.nan

        value = path_length / displacement
        return (
            float(value)
            if np.isfinite(value)
            else np.nan
        )

    def _compute_subject_switch_geometry(
            self,
            exp,
            switch_type,
            latent_key="latents",
            rt_col="mrt_ms",
            correct_only=True,
            verbose=False,
    ):
        """Average switch-trajectory metrics equally across stimulus cells."""
        rows = []

        for stim_gender in [0, 1]:
            for stim_emotion in [0, 1]:
                try:
                    rows.append(
                        self._compute_condition_geometry(
                            exp=exp,
                            switch_type=switch_type,
                            stim_gender=stim_gender,
                            stim_emotion=stim_emotion,
                            latent_key=latent_key,
                            rt_col=rt_col,
                            correct_only=correct_only,
                        )
                    )
                except ValueError as error:
                    if verbose:
                        print(
                            "[Warning] {}: skipping gender={}, emotion={}: "
                            "{}".format(
                                switch_type,
                                stim_gender,
                                stim_emotion,
                                error,
                            )
                        )

        if len(rows) == 0:
            raise ValueError(
                "No valid stimulus conditions found for {}.".format(
                    switch_type
                )
            )

        condition_df = pd.DataFrame(
            rows
        )

        def condition_mean(column):
            values = condition_df[
                column
            ].to_numpy(
                dtype=float
            )
            valid = np.isfinite(values)

            if not np.any(valid):
                return np.nan

            return float(
                np.mean(
                    values[valid]
                )
            )

        summary = {
            "switch_type": switch_type,
            "n_conditions": int(
                len(condition_df)
            ),
            "n_switch_total": int(
                condition_df["n_switch"].sum()
            ),
            "n_reference_total": int(
                condition_df["n_reference"].sum()
            ),
            "bar_L_0_tc": condition_mean(
                "L_0_tc"
            ),
            "bar_L_tc_rt": condition_mean(
                "L_tc_rt"
            ),
            "bar_L_0_rt": condition_mean(
                "L_0_rt"
            ),
            "bar_D_0_tc": condition_mean(
                "D_0_tc"
            ),
            "bar_D_tc_rt": condition_mean(
                "D_tc_rt"
            ),
            "bar_D_0_rt": condition_mean(
                "D_0_rt"
            ),
            "bar_F_0_tc": condition_mean(
                "F_0_tc"
            ),
            "bar_F_tc_rt": condition_mean(
                "F_tc_rt"
            ),
            "bar_F_0_rt": condition_mean(
                "F_0_rt"
            ),
            "bar_tc_ms": condition_mean(
                "tc_ms"
            ),
            "bar_rt_ms_switch": condition_mean(
                "mean_rt_ms_switch"
            ),
            "bar_rt_minus_tc_ms": condition_mean(
                "rt_minus_tc_ms"
            ),
            "bar_min_dist_to_reference": condition_mean(
                "min_dist_to_reference"
            ),
        }

        return summary, condition_df

    def _compute_group_geometry_hddm_dataframe(self):
        """Build the participant-level geometry/HDDM dataframe."""
        rows = []

        for switch_type in ("E2G", "G2E"):
            for patient_id in self.patients_id:
                exp = self.subj_exps[
                    patient_id
                ]

                try:
                    summary, _ = (
                        self._compute_subject_switch_geometry(
                            exp=exp,
                            switch_type=switch_type,
                            latent_key="latents",
                            rt_col="mrt_ms",
                            correct_only=True,
                            verbose=True,
                        )
                    )
                except ValueError as error:
                    print(
                        "[Warning] Skipping subj={}, switch_type={}: "
                        "{}".format(
                            patient_id,
                            switch_type,
                            error,
                        )
                    )
                    continue

                hddm_values = self.hddm_by_subject[
                    patient_id
                ][switch_type]

                row = {
                    "subj_id": patient_id,
                    **summary,
                    "hddm_drift_rate": float(
                        hddm_values["drift_rate"]
                    ),
                    "hddm_nondecision_time": float(
                        hddm_values[
                            "nondecision_time"
                        ]
                    ),
                }
                rows.append(row)

        return pd.DataFrame(
            rows
        )

    @staticmethod
    def _sem(values):
        """Return SEM after removing non-finite values."""
        values = np.asarray(
            values,
            dtype=float,
        )
        values = values[
            np.isfinite(values)
        ]

        if len(values) < 2:
            return np.nan

        return np.nanstd(
            values,
            ddof=1,
        ) / np.sqrt(
            len(values)
        )

    @staticmethod
    def _bootstrap_regression_band(
            x,
            y,
            plot_x,
            n_boot=10000,
            ci=95,
            seed=0,
    ):
        """Bootstrap a pointwise confidence band for a linear regression."""
        rng = np.random.default_rng(
            seed
        )

        x = np.asarray(
            x,
            dtype=float,
        )
        y = np.asarray(
            y,
            dtype=float,
        )

        valid = (
                np.isfinite(x)
                & np.isfinite(y)
        )
        x = x[valid]
        y = y[valid]

        slope, intercept = np.polyfit(
            x,
            y,
            1,
        )
        y_hat = (
                slope * plot_x
                + intercept
        )

        boot_lines = []
        n_samples = len(x)

        for _ in range(n_boot):
            indices = rng.integers(
                0,
                n_samples,
                size=n_samples,
            )
            x_boot = x[indices]
            y_boot = y[indices]

            if np.nanstd(x_boot) == 0:
                continue

            slope_boot, intercept_boot = np.polyfit(
                x_boot,
                y_boot,
                1,
            )
            boot_lines.append(
                slope_boot * plot_x
                + intercept_boot
            )

        boot_lines = np.asarray(
            boot_lines
        )
        alpha = (
                        100 - ci
                ) / 2.0

        y_low = np.nanpercentile(
            boot_lines,
            alpha,
            axis=0,
        )
        y_high = np.nanpercentile(
            boot_lines,
            100 - alpha,
            axis=0,
        )

        return y_hat, y_low, y_high

    def _plot_tc_hddm_panel(
            self,
            ax,
            df,
            switch_type,
            hddm_param,
            geom_metric,
            color,
            line_ext_frac=0.05,
            plot_boot_band=True,
            n_boot_band=10000,
            ci=95,
            seed=0,
            scatter_size=14,
            range_xlim=None,
            range_ylim=None,
    ):
        """Plot one trajectory-geometry/HDDM correlation panel."""
        subset = df.loc[
            df["switch_type"] == switch_type
            ].copy()

        x = subset[
            geom_metric
        ].to_numpy(
            dtype=float
        )
        y = subset[
            hddm_param
        ].to_numpy(
            dtype=float
        )

        valid = (
                np.isfinite(x)
                & np.isfinite(y)
        )
        x = x[valid]
        y = y[valid]

        if len(x) < 3:
            raise ValueError(
                "Not enough valid data for switch_type={}, "
                "hddm_param={}, geom_metric={}.".format(
                    switch_type,
                    hddm_param,
                    geom_metric,
                )
            )

        if hddm_param == "hddm_nondecision_time":
            y_label = "HDDM non-decision time $t_0$ (s)"
            parameter_name = "$t_0$"
        elif hddm_param == "hddm_drift_rate":
            y_label = "HDDM drift rate $v$"
            parameter_name = "$v$"
        else:
            y_label = hddm_param
            parameter_name = hddm_param

        metric_labels = {
            "bar_F_0_tc": (
                r"Early trajectory tortuosity $\bar{T}_{[0,t_c]}$",
                r"$\bar{T}_{[0,t_c]}$",
            ),
            "bar_F_tc_rt": (
                r"Late trajectory tortuosity $\bar{T}_{[t_c,RT]}$",
                r"$\bar{T}_{[t_c,RT]}$",
            ),
        }
        x_label, metric_name = metric_labels.get(
            geom_metric,
            (geom_metric, geom_metric),
        )

        x_range = (
                np.nanmax(x)
                - np.nanmin(x)
        )
        pad = (
            line_ext_frac * x_range
            if x_range > 0
            else 1.0
        )
        plot_x = np.array(
            [
                np.nanmin(x) - pad,
                np.nanmax(x) + pad,
            ]
        )

        slope, intercept = np.polyfit(
            x,
            y,
            1,
        )

        ax.plot(
            plot_x,
            slope * plot_x + intercept,
            "k-",
            zorder=2,
            linewidth=0.75,
        )
        ax.scatter(
            x,
            y,
            s=scatter_size,
            marker="o",
            zorder=3,
            color=color,
            edgecolor="none",
            alpha=0.9,
        )

        if plot_boot_band:
            y_hat, y_low, y_high = (
                self._bootstrap_regression_band(
                    x,
                    y,
                    plot_x,
                    n_boot=n_boot_band,
                    ci=ci,
                    seed=seed,
                )
            )
            ax.plot(
                plot_x,
                y_hat,
                linewidth=1.0,
                color="r",
                zorder=4,
            )
            ax.fill_between(
                plot_x,
                y_low,
                y_high,
                alpha=0.2,
                color="grey",
                zorder=1,
            )

        pearson_r, pearson_p = pearsonr(
            x,
            y,
        )
        ci_low, ci_high = self._bootstrap_pearson_ci(
            x,
            y,
            n_boot=n_boot_band,
            ci=ci,
            seed=seed,
        )

        print(
            "{}: {} vs. {}".format(
                switch_type,
                parameter_name,
                metric_name,
            )
        )
        print(
            "Pearson: r = {:.3f}, {:.0f}% CI = "
            "[{:.3f}, {:.3f}], p = {:.3e} ({})".format(
                pearson_r,
                ci,
                ci_low,
                ci_high,
                pearson_p,
                p_to_stars(
                    pearson_p
                ),
            )
        )
        print(
            "Best-fit slope: {:.6f}; intercept: {:.6f}".format(
                slope,
                intercept,
            )
        )
        print(
            "Mean +/- s.e.m. {}: {:.4f} +/- {:.4f}".format(
                metric_name,
                np.nanmean(x),
                self._sem(x),
            )
        )
        print(
            "Mean +/- s.e.m. {}: {:.4f} +/- {:.4f}".format(
                parameter_name,
                np.nanmean(y),
                self._sem(y),
            )
        )
        print("-----------------------------")

        ax.set_xlabel(
            x_label
        )
        ax.set_ylabel(
            y_label
        )
        ax.set_title(
            "{}: {} vs. {}".format(
                switch_type,
                parameter_name,
                metric_name,
            )
        )
        ax.text(
            0.05,
            0.95,
            "Pearson r = {:.2f}, {}\n".format(
                pearson_r,
                p_to_stars(
                    pearson_p
                ),
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )

        if range_xlim is not None:
            ax.set_xlim(
                range_xlim
            )
        if range_ylim is not None:
            ax.set_ylim(
                range_ylim
            )

        return {
            "switch_type": switch_type,
            "hddm_param": hddm_param,
            "geom_metric": geom_metric,
            "n": len(x),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "pearson_ci_low": ci_low,
            "pearson_ci_high": ci_high,
            "slope": slope,
            "intercept": intercept,
        }

    @staticmethod
    def _bootstrap_pearson_ci(
            x,
            y,
            n_boot=10000,
            ci=95,
            seed=0,
    ):
        """Return a bootstrap confidence interval for Pearson's r."""
        rng = np.random.default_rng(
            seed
        )
        n_samples = len(x)
        boot_rs = []

        for _ in range(n_boot):
            indices = rng.integers(
                0,
                n_samples,
                size=n_samples,
            )
            x_boot = x[indices]
            y_boot = y[indices]

            if (
                    np.nanstd(x_boot) == 0
                    or np.nanstd(y_boot) == 0
            ):
                continue

            r_boot, _ = pearsonr(
                x_boot,
                y_boot,
            )
            if np.isfinite(r_boot):
                boot_rs.append(
                    r_boot
                )

        boot_rs = np.asarray(
            boot_rs,
            dtype=float,
        )
        if len(boot_rs) == 0:
            return np.nan, np.nan

        alpha = (
                        100 - ci
                ) / 2.0
        ci_low = float(
            np.nanpercentile(
                boot_rs,
                alpha,
            )
        )
        ci_high = float(
            np.nanpercentile(
                boot_rs,
                100 - alpha,
            )
        )

        return ci_low, ci_high

    def _plot_tc_hddm_four_panels(
            self,
            axes,
            df_geom_hddm,
            colors=None,
            plot_boot_band=True,
            n_boot_band=10000,
            ci=95,
            seed=0,
    ):
        """Plot the four primary trajectory-tortuosity/HDDM correlations."""
        if colors is None:
            colors = {
                "E2G": Constants.COLOR_E2G,
                "G2E": Constants.COLOR_G2E,
            }

        panel_specs = [
            {
                "switch_type": "E2G",
                "hddm_param": "hddm_nondecision_time",
                "geom_metric": "bar_F_0_tc",
                "color": colors["E2G"],
                "seed": seed,
                "range_xlim": [1.2, 2.5],
                "range_ylim": [0.5, 1.8],
            },
            {
                "switch_type": "G2E",
                "hddm_param": "hddm_nondecision_time",
                "geom_metric": "bar_F_0_tc",
                "color": colors["G2E"],
                "seed": seed + 1,
                "range_xlim": [1.2, 2.5],
                "range_ylim": [0.5, 1.8],
            },
            {
                "switch_type": "E2G",
                "hddm_param": "hddm_drift_rate",
                "geom_metric": "bar_F_tc_rt",
                "color": colors["E2G"],
                "seed": seed,
                "range_xlim": [0.8, 6.8],
                "range_ylim": [-0.03, 0.25],
            },
            {
                "switch_type": "G2E",
                "hddm_param": "hddm_drift_rate",
                "geom_metric": "bar_F_tc_rt",
                "color": colors["G2E"],
                "seed": seed,
                "range_xlim": [0.8, 6.8],
                "range_ylim": [-0.03, 0.25],
            },
        ]

        results = []
        for ax, spec in zip(
                axes,
                panel_specs,
        ):
            results.append(
                self._plot_tc_hddm_panel(
                    ax=ax,
                    df=df_geom_hddm,
                    switch_type=spec["switch_type"],
                    hddm_param=spec["hddm_param"],
                    geom_metric=spec["geom_metric"],
                    color=spec["color"],
                    plot_boot_band=plot_boot_band,
                    n_boot_band=n_boot_band,
                    ci=ci,
                    seed=spec["seed"],
                    range_xlim=spec["range_xlim"],
                    range_ylim=spec["range_ylim"],
                )
            )

        # Preserve the original panel lettering within this figure module.
        for ax, label in zip(
                axes,
                ["a", "b", "c", "d"],
        ):
            ax.text(
                -0.22,
                1.08,
                label,
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                va="top",
            )

        return pd.DataFrame(
            results
        )

    @staticmethod
    def _print_holm_corrections(stats_df):
        """Apply the two planned two-test Holm correction families."""
        correction_specs = [
            (
                "Panel c",
                "hddm_nondecision_time",
                ["E2G", "G2E"],
            ),
            (
                "Panel d",
                "hddm_drift_rate",
                ["E2G", "G2E"],
            ),
        ]

        for panel_name, parameter, switch_types in correction_specs:
            raw_pvalues = []
            for switch_type in switch_types:
                p_value = stats_df.loc[
                    (stats_df["switch_type"] == switch_type)
                    & (stats_df["hddm_param"] == parameter),
                    "pearson_p",
                ].iloc[0]
                raw_pvalues.append(
                    p_value
                )

            reject, adjusted_pvalues, _, _ = multipletests(
                raw_pvalues,
                alpha=0.05,
                method="holm",
            )

            for (
                    switch_type,
                    raw_p,
                    adjusted_p,
                    significant,
            ) in zip(
                switch_types,
                raw_pvalues,
                adjusted_pvalues,
                reject,
            ):
                print(
                    "{} {} raw p = {} Holm p = {} "
                    "significant = {}".format(
                        panel_name,
                        switch_type,
                        raw_p,
                        adjusted_p,
                        significant,
                    )
                )

    def _make_transition_metric_boxplot(
            self,
            df,
            ax,
            metric,
            metric_label,
    ):
        """Compare one trajectory metric between E2G and G2E."""
        order = [
            "E2G",
            "G2E",
        ]
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
                ["subj_id", "switch_type"],
                as_index=False,
            )[metric]
            .mean()
        )

        summary = (
            df_subject.groupby(
                "switch_type"
            )[metric]
            .agg(
                mean_metric="mean",
                sd_metric="std",
                n="count",
            )
            .reset_index()
        )
        summary["sem_metric"] = (
                summary["sd_metric"]
                / np.sqrt(
            summary["n"]
        )
        )

        print(
            "{} summary statistics:".format(
                metric_label
            )
        )
        print(
            summary
        )

        sns.boxplot(
            data=df_subject,
            x="switch_type",
            y=metric,
            order=order,
            palette=palette,
            showcaps=True,
            fliersize=0,
            width=0.7,
            linewidth=0.5,
            ax=ax,
        )
        sns.stripplot(
            data=df_subject,
            x="switch_type",
            y=metric,
            order=order,
            palette=palette,
            alpha=0.7,
            size=4,
            jitter=False,
            zorder=10,
            linewidth=0.2,
            ax=ax,
        )

        for _, group in df_subject.groupby(
                "subj_id",
                sort=False,
        ):
            if set(
                    group["switch_type"]
            ) >= set(order):
                xs = [
                    xpos[transition]
                    for transition in order
                ]
                ys = [
                    group.loc[
                        group["switch_type"] == transition,
                        metric,
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
            df_subject[metric].min()
        )
        y_max_data = float(
            df_subject[metric].max()
        )
        y_span = (
            y_max_data - y_min_data
            if y_max_data - y_min_data > 1e-9
            else 1.0
        )

        y_top = (
                y_max_data
                + 0.25 * y_span
        )
        ax.set_ylim(
            y_min_data - 0.08 * y_span,
            y_top,
        )

        y_min_ax, y_max_ax = ax.get_ylim()
        y_span_ax = (
                y_max_ax - y_min_ax
        )
        bracket_y = (
                y_max_data
                + 0.08 * y_span_ax
        )
        bracket_h = (
                0.015 * y_span_ax
        )
        text_pad = (
                0.001 * y_span_ax
        )

        max_text_y = (
                y_top
                - 0.01 * y_span_ax
        )
        bracket_y = min(
            bracket_y,
            max_text_y
            - (
                    bracket_h
                    + text_pad
            ),
        )

        wide = df_subject.pivot_table(
            index="subj_id",
            columns="switch_type",
            values=metric,
            aggfunc="mean",
        )
        paired = wide[
            order
        ].dropna()

        e2g_values = paired[
            "E2G"
        ].values
        g2e_values = paired[
            "G2E"
        ].values

        t_stat, p_value = ttest_rel(
            e2g_values,
            g2e_values,
            nan_policy="omit",
        )
        difference = (
                e2g_values
                - g2e_values
        )

        print(
            "{} paired t-test:".format(
                metric_label
            )
        )
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
                np.mean(
                    difference
                ),
                sem(
                    difference,
                    nan_policy="omit",
                ),
            )
        )
        print(
            "E2G mean +/- s.e.m. = {} +/- {}".format(
                np.mean(
                    e2g_values
                ),
                sem(
                    e2g_values,
                    nan_policy="omit",
                ),
            )
        )
        print(
            "G2E mean +/- s.e.m. = {} +/- {}".format(
                np.mean(
                    g2e_values
                ),
                sem(
                    g2e_values,
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
            text=p_to_stars(
                p_value
            ),
            text_pad=text_pad,
            lw=1.0,
            fontsize=8,
        )

        ax.set_xlabel(
            "Switch transition"
        )
        ax.set_ylabel(
            metric_label
        )

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
