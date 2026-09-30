"""Generate Figure 3b-e: semantic-axis geometry and orthogonality."""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.multitest import multipletests

from affective_task.utils import (
    Constants,
    compute_aligned_projections_joint,
    compute_irrelevant_semantic_axes_raw,
    save_figure,
)


class Figure3BCDE:
    """Plot semantic-axis angular structure and orthogonality statistics."""

    analysis_dir = "model_analysis"
    stats_fn = "holdout_outputs_01SD.pkl"
    fp_fn = "fixed_points.pkl"

    figsize = (9, 9)
    figdpi = 300

    def __init__(self, model_dir, save_dir, patients_id, rand_seed, n_boot):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.patients_id = patients_id

        # Retain the common figure-class constructor signature.
        _ = rand_seed
        _ = n_boot

        # Directional one-sided tests use alpha = 0.025.
        self.alpha_one_sided = 0.025
        self.t_start_ind = 5

        self.group_expt_repeat = {}
        self.group_fp = {}

    def make_figure(self):
        """Generate and save Figure 3b-e."""
        print("Making Figure 3b-e...")
        self._run_preprocessing()

        print("Stats for Figure 3b-e")
        print("---------------------")

        fig = self._plot_figure_get_stats()
        save_figure(fig, self.save_dir, "Fig3bcde")
        return fig

    def _run_preprocessing(self):
        """Load subject data and derive relevant/irrelevant semantic axes."""
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

            fp_path = os.path.join(
                subject_dir,
                self.analysis_dir,
                self.fp_fn,
            )
            with open(fp_path, "rb") as path:
                fixed_points = pickle.load(path)

            self.group_expt_repeat[str(patient_id)] = expt_stats
            self.group_fp[str(patient_id)] = fixed_points

        rows = []
        for patient_id, fixed_points in self.group_fp.items():
            expt_stats = self.group_expt_repeat[str(patient_id)]
            latents = expt_stats.windowed["umap_latents"][:, :, :]

            for task_cue in [0, 1]:
                for stim_gender in [0, 1]:
                    for stim_emotion in [0, 1]:
                        fixed_point = fixed_points.loc[
                            (fixed_points["cue"] == task_cue)
                            & (fixed_points["stim_gender"] == stim_gender)
                            & (fixed_points["stim_emotion"] == stim_emotion),
                            "zloc_umap",
                        ].values[0]

                        condition_filter = {
                            "task_cue": task_cue,
                            "prev_task_cue": task_cue,
                            "stim_gender": stim_gender,
                            "stim_emotion": stim_emotion,
                        }
                        filtered_inds = expt_stats.select(**condition_filter)

                        x = np.mean(
                            latents[:, filtered_inds, 0],
                            axis=1,
                        )
                        y = np.mean(
                            latents[:, filtered_inds, 1],
                            axis=1,
                        )
                        z = np.mean(
                            latents[:, filtered_inds, 2],
                            axis=1,
                        )

                        task = "gender" if task_cue == 0 else "emotion"
                        gender = (
                            "male"
                            if stim_gender == 0
                            else "female"
                        )
                        emotion = (
                            "happy"
                            if stim_emotion == 0
                            else "threatening"
                        )

                        rows.append(
                            [
                                str(patient_id),
                                task,
                                gender,
                                emotion,
                                x[self.t_start_ind],
                                y[self.t_start_ind],
                                z[self.t_start_ind],
                                fixed_point[0],
                                fixed_point[1],
                                fixed_point[2],
                            ]
                        )

        group_representation = pd.DataFrame(
            rows,
            columns=[
                "patient",
                "task",
                "gender",
                "emotion",
                "x",
                "y",
                "z",
                "fix_x",
                "fix_y",
                "fix_z",
            ],
        )

        (
            _,
            _,
            self.group_diags,
            _,
            self.axes_raw_relevant,
        ) = compute_aligned_projections_joint(group_representation)

        self._print_basis_diagnostics()

        self.axes_raw_irrelevant = compute_irrelevant_semantic_axes_raw(
            group_representation
        )

    def _print_basis_diagnostics(self):
        """Print Gram-Schmidt fallback diagnostics for the semantic basis."""
        degenerate_gender = [
            str(subject_id)
            for subject_id, diagnostics in self.group_diags.items()
            if diagnostics.get("gender_degenerate", False)
        ]
        gender_fallback = [
            str(subject_id)
            for subject_id, diagnostics in self.group_diags.items()
            if diagnostics.get("gender_crossprod_fallback", False)
        ]
        emotion_fallback = [
            str(subject_id)
            for subject_id, diagnostics in self.group_diags.items()
            if diagnostics.get("emotion_crossprod_fallback", False)
        ]

        print("=" * 80)
        print("Semantic-basis Gram-Schmidt diagnostics")
        print(
            "Degenerate gender residuals: {}/{}".format(
                len(degenerate_gender),
                len(self.group_diags),
            )
        )
        print("Degenerate gender subjects:", degenerate_gender)
        print(
            "Gender cross-product fallbacks: {}/{}".format(
                len(gender_fallback),
                len(self.group_diags),
            )
        )
        print("Gender fallback subjects:", gender_fallback)
        print(
            "Emotion cross-product fallbacks: {}/{}".format(
                len(emotion_fallback),
                len(self.group_diags),
            )
        )
        print("Emotion fallback subjects:", emotion_fallback)
        print("=" * 80)

    def _plot_figure_get_stats(self):
        """Assemble Figure 3b-e and print the associated statistics."""
        fig = plt.figure(
            constrained_layout=False,
            figsize=self.figsize,
            dpi=self.figdpi,
        )
        gs = fig.add_gridspec(14, 14)

        # ------------------------------------------------------------------
        # Figure 3b: angular relationships among task-relevant axes
        # ------------------------------------------------------------------
        ax_relevant = fig.add_subplot(
            gs[0:5, 0:5],
            projection="polar",
        )
        relevant_stats = self.plot_orthogonality_rose_combined(
            ax=ax_relevant,
            semantic_axes_raw=self.axes_raw_relevant,
            unsigned=False,
            mu0_deg=90.0,
            bins=18,
            n_perm=20000,
            seed=0,
            fold_for_display=False,
            histtype="bar",
            alpha=0.35,
        )
        print(relevant_stats)

        axis_pairs = [
            "Task-Gender",
            "Task-Emotion",
            "Gender-Emotion",
        ]
        raw_pvalues = [
            relevant_stats[pair]["p"]
            for pair in axis_pairs
        ]
        reject, holm_pvalues, _, _ = multipletests(
            raw_pvalues,
            alpha=self.alpha_one_sided,
            method="holm",
        )

        for pair, p_raw, p_corr, significant in zip(
                axis_pairs,
                raw_pvalues,
                holm_pvalues,
                reject,
        ):
            print(
                pair,
                "raw p =",
                p_raw,
                "Holm p =",
                p_corr,
                "one-sided alpha =",
                self.alpha_one_sided,
                "significant =",
                significant,
            )

        # ------------------------------------------------------------------
        # Figure 3c: relevant vs. irrelevant semantic-axis alignment
        # ------------------------------------------------------------------
        ax_rel_irr = fig.add_subplot(
            gs[0:5, 9:14],
            projection="polar",
        )
        rel_irr_stats = self.plot_rel_vs_irr_axis_rose_combined(
            ax=ax_rel_irr,
            axes_raw_relevant=self.axes_raw_relevant,
            axes_raw_irrelevant=self.axes_raw_irrelevant,
            unsigned=False,
            mu0_deg=0.0,
            bins=18,
            n_perm=20000,
            seed=0,
            fold_for_display=False,
            alpha=0.35,
        )
        print(rel_irr_stats)

        for comparison, result in rel_irr_stats.items():
            if result is not None:
                print(
                    comparison,
                    "raw p =",
                    result["p"],
                    "one-sided alpha =",
                    self.alpha_one_sided,
                    "significant =",
                    result["p"] < self.alpha_one_sided,
                )

        # ------------------------------------------------------------------
        # Figure 3d-e: Gram matrix and matched random-triad null
        # ------------------------------------------------------------------
        df_orth, gram_by_subject = (
            self.compute_task_relevant_orthogonality_metrics(
                self.axes_raw_relevant
            )
        )

        ax_gram = fig.add_subplot(gs[9:14, 0:5])
        ax_null = fig.add_subplot(gs[9:14, 9:14])

        self.plot_task_relevant_orthogonality_summary(
            axes=[ax_gram, ax_null],
            df_orth=df_orth,
            gram_by_subject=gram_by_subject,
            n_null=10000,
            seed=0,
            ambient_dim=3,
            hist_bins=20,
        )

        return fig

    def plot_task_relevant_orthogonality_summary(
            self,
            axes,
            df_orth,
            gram_by_subject,
            n_null=10000,
            seed=0,
            ambient_dim=3,
            hist_bins=30,
    ):
        """Plot the mean Gram matrix and matched random-triad null.

        Each null sample is the mean orthogonality error from the same number
        of independently generated random triads as there are observed
        subjects. The one-sided Monte Carlo p-value tests whether the observed
        group mean is smaller than expected under this null.
        """
        subject_ids = list(gram_by_subject.keys())
        gram_stack = np.stack(
            [
                gram_by_subject[subject_id]
                for subject_id in subject_ids
            ],
            axis=0,
        )
        gram_mean = np.nanmean(gram_stack, axis=0)

        observed = (
            pd.to_numeric(
                df_orth["orth_err_fro"],
                errors="coerce",
            )
            .dropna()
            .to_numpy(dtype=float)
        )
        n_observed = len(observed)
        if n_observed == 0:
            raise ValueError(
                "No valid observed orthogonality errors were found."
            )

        observed_mean = float(np.mean(observed))

        rng = np.random.RandomState(seed)
        null_group_means = np.empty(
            n_null,
            dtype=float,
        )

        for i in range(n_null):
            null_errors = np.empty(
                n_observed,
                dtype=float,
            )
            for j in range(n_observed):
                random_axes = rng.normal(
                    size=(ambient_dim, 3)
                )
                random_axes = (
                        random_axes
                        / np.linalg.norm(
                    random_axes,
                    axis=0,
                    keepdims=True,
                )
                )
                gram = random_axes.T @ random_axes
                null_errors[j] = np.linalg.norm(
                    gram - np.eye(3),
                    ord="fro",
                )

            null_group_means[i] = np.mean(null_errors)

        p_less = (
                         np.sum(null_group_means <= observed_mean) + 1.0
                 ) / (n_null + 1.0)
        significant = p_less < self.alpha_one_sided

        null_mean = float(np.mean(null_group_means))
        null_q025 = float(
            np.percentile(null_group_means, 2.5)
        )
        null_q975 = float(
            np.percentile(null_group_means, 97.5)
        )

        # Mean Gram matrix.
        gram_ax = axes[0]
        image = gram_ax.imshow(
            gram_mean,
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
        )
        labels = ["Task", "Emotion", "Gender"]
        gram_ax.set_xticks(range(3))
        gram_ax.set_yticks(range(3))
        gram_ax.set_xticklabels(labels)
        gram_ax.set_yticklabels(labels)
        gram_ax.set_title("Mean Gram matrix")

        for i in range(3):
            for j in range(3):
                gram_ax.text(
                    j,
                    i,
                    "{:.2f}".format(gram_mean[i, j]),
                    ha="center",
                    va="center",
                    fontsize=10,
                )

        colorbar = plt.colorbar(
            image,
            ax=gram_ax,
            fraction=0.046,
            pad=0.04,
        )
        colorbar.set_label(r"$E^\top E$ value")

        # Matched group-mean random-triad null.
        null_ax = axes[1]
        null_ax.hist(
            null_group_means,
            bins=hist_bins,
            density=True,
            alpha=0.25,
            edgecolor="none",
        )
        sns.kdeplot(
            x=null_group_means,
            ax=null_ax,
            fill=False,
            linewidth=1.8,
        )

        null_ax.axvline(
            observed_mean,
            linestyle="-",
            linewidth=2.0,
        )
        null_ax.axvline(
            null_mean,
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
        null_ax.axvline(
            null_q025,
            linestyle=":",
            linewidth=1.0,
            alpha=0.8,
        )
        null_ax.axvline(
            null_q975,
            linestyle=":",
            linewidth=1.0,
            alpha=0.8,
        )

        ymin, ymax = null_ax.get_ylim()
        rug_y = np.full(
            len(observed),
            ymin + 0.02 * (ymax - ymin),
        )
        null_ax.scatter(
            observed,
            rug_y,
            s=14,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.3,
            zorder=3,
        )

        null_ax.set_xlabel(
            r"Group-mean orthogonality error "
            r"$\|E^\top E - I\|_F$"
        )
        null_ax.set_ylabel("Density")
        null_ax.set_title(
            "Matched random-triad null ({}D)".format(
                ambient_dim
            )
        )
        null_ax.grid(axis="x", alpha=0.25)

        annotation = (
            "Observed mean = {:.3f}\n"
            "Null mean = {:.3f}\n"
            "95% null = [{:.3f}, {:.3f}]\n"
            "one-sided p = {:.5f}\n"
            "alpha = {:.3f}; significant = {}".format(
                observed_mean,
                null_mean,
                null_q025,
                null_q975,
                p_less,
                self.alpha_one_sided,
                significant,
            )
        )
        null_ax.text(
            0.98,
            0.98,
            annotation,
            transform=null_ax.transAxes,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round",
                "fc": "white",
                "ec": "0.7",
                "alpha": 0.9,
            },
            fontsize=9,
        )

        print(
            "p_less",
            p_less,
            "one-sided alpha",
            self.alpha_one_sided,
            "significant",
            significant,
        )

    @staticmethod
    def _unit(vector, eps=1e-12):
        """Return a unit-normalized vector or NaNs for a degenerate vector."""
        vector = np.asarray(
            vector,
            dtype=float,
        )
        norm = np.linalg.norm(vector)

        if not np.isfinite(norm) or norm < eps:
            return np.full_like(
                vector,
                np.nan,
            )

        return vector / norm

    def compute_task_relevant_orthogonality_metrics(
            self,
            axes_raw_relevant,
    ):
        """Compute subject-level Gram matrices and orthogonality metrics."""
        rows = []
        gram_by_subject = {}

        for subject_id, axes in axes_raw_relevant.items():
            task_axis = self._unit(axes["task"])
            emotion_axis = self._unit(axes["emotion"])
            gender_axis = self._unit(axes["gender"])

            basis = np.column_stack(
                [
                    task_axis,
                    emotion_axis,
                    gender_axis,
                ]
            )
            gram = basis.T @ basis
            gram_by_subject[subject_id] = gram

            cos_task_emotion = gram[0, 1]
            cos_task_gender = gram[0, 2]
            cos_emotion_gender = gram[1, 2]

            angle_task_emotion = np.degrees(
                np.arccos(
                    np.clip(
                        cos_task_emotion,
                        -1.0,
                        1.0,
                    )
                )
            )
            angle_task_gender = np.degrees(
                np.arccos(
                    np.clip(
                        cos_task_gender,
                        -1.0,
                        1.0,
                    )
                )
            )
            angle_emotion_gender = np.degrees(
                np.arccos(
                    np.clip(
                        cos_emotion_gender,
                        -1.0,
                        1.0,
                    )
                )
            )

            orth_error = np.linalg.norm(
                gram - np.eye(3),
                ord="fro",
            )
            off_diagonal = np.abs(
                [
                    cos_task_emotion,
                    cos_task_gender,
                    cos_emotion_gender,
                ]
            )
            mean_abs_offdiag = np.mean(off_diagonal)
            max_abs_offdiag = np.max(off_diagonal)

            gram_det = max(
                np.linalg.det(gram),
                0.0,
            )
            abs_det_like = np.sqrt(gram_det)

            rows.append(
                {
                    "patient": subject_id,
                    "cos_task_emotion": cos_task_emotion,
                    "cos_task_gender": cos_task_gender,
                    "cos_emotion_gender": cos_emotion_gender,
                    "angle_task_emotion": angle_task_emotion,
                    "angle_task_gender": angle_task_gender,
                    "angle_emotion_gender": angle_emotion_gender,
                    "orth_err_fro": orth_error,
                    "mean_abs_offdiag": mean_abs_offdiag,
                    "max_abs_offdiag": max_abs_offdiag,
                    "abs_det_like": abs_det_like,
                }
            )

        return pd.DataFrame(rows), gram_by_subject

    def _collect_rel_vs_irr_pair_angles(
            self,
            axes_raw_relevant,
            axes_raw_irrelevant,
            unsigned=False,
    ):
        """Collect relevant-vs-irrelevant gender/emotion angles."""
        pair_angles = {
            "Gender relevant vs irrelevant": [],
            "Emotion relevant vs irrelevant": [],
        }

        common_subjects = sorted(
            set(axes_raw_relevant.keys())
            & set(axes_raw_irrelevant.keys())
        )

        for subject_id in common_subjects:
            relevant = axes_raw_relevant[subject_id]
            irrelevant = axes_raw_irrelevant[subject_id]

            if (
                    "gender" in relevant
                    and "gender" in irrelevant
            ):
                pair_angles[
                    "Gender relevant vs irrelevant"
                ].append(
                    self._angle_deg(
                        relevant["gender"],
                        irrelevant["gender"],
                        unsigned=unsigned,
                    )
                )

            if (
                    "emotion" in relevant
                    and "emotion" in irrelevant
            ):
                pair_angles[
                    "Emotion relevant vs irrelevant"
                ].append(
                    self._angle_deg(
                        relevant["emotion"],
                        irrelevant["emotion"],
                        unsigned=unsigned,
                    )
                )

        return pair_angles

    def plot_rel_vs_irr_axis_rose_combined(
            self,
            ax,
            axes_raw_relevant,
            axes_raw_irrelevant,
            unsigned=False,
            mu0_deg=0.0,
            bins=18,
            n_perm=20000,
            seed=0,
            fold_for_display=False,
            rmax=None,
            alpha=0.35,
            colors=None,
            show_mean_lines=True,
            show_text=True,
    ):
        """Plot relevant-vs-irrelevant gender/emotion angle distributions."""
        pair_angles = self._collect_rel_vs_irr_pair_angles(
            axes_raw_relevant=axes_raw_relevant,
            axes_raw_irrelevant=axes_raw_irrelevant,
            unsigned=unsigned,
        )

        support_deg = 90.0 if unsigned else 180.0
        titles = [
            "Gender relevant vs irrelevant",
            "Emotion relevant vs irrelevant",
        ]

        if colors is None:
            colors = {
                "Gender relevant vs irrelevant": Constants.COLOR_G2G,
                "Emotion relevant vs irrelevant": Constants.COLOR_E2E,
            }

        if fold_for_display and support_deg > 90.0:
            plot_support_deg = 90.0
        else:
            plot_support_deg = float(support_deg)

        edges = np.linspace(
            0.0,
            np.deg2rad(plot_support_deg),
            bins + 1,
        )
        widths = np.diff(edges)
        centers = edges[:-1] + widths / 2.0

        results = {}
        all_counts = []

        for i, title in enumerate(titles):
            angles = np.asarray(
                pair_angles[title],
                dtype=float,
            )
            angles = angles[np.isfinite(angles)]

            if len(angles) == 0:
                results[title] = None
                continue

            vtest = self._v_test_perm(
                angles_deg=angles,
                mu0_deg=mu0_deg,
                n_perm=n_perm,
                seed=seed + i,
                support_deg=support_deg,
            )
            results[title] = vtest

            plot_angles = angles.copy()
            if fold_for_display and support_deg > 90.0:
                folded = np.mod(
                    plot_angles,
                    support_deg,
                )
                plot_angles = np.where(
                    folded > 90.0,
                    180.0 - folded,
                    folded,
                )

            counts, _ = np.histogram(
                np.deg2rad(plot_angles),
                bins=edges,
            )
            all_counts.append(counts)

            color = colors[title]
            ax.bar(
                centers,
                counts,
                width=widths,
                bottom=0.0,
                align="center",
                color=color,
                alpha=alpha,
                edgecolor=color,
                linewidth=0.8,
                label=title,
            )

            if show_mean_lines:
                mean_direction = vtest["mu_deg"]
                if (
                        fold_for_display
                        and support_deg > 90.0
                ):
                    mean_direction %= 180.0
                    if mean_direction > 90.0:
                        mean_direction = (
                                180.0 - mean_direction
                        )

                local_rmax = max(
                    1.0,
                    float(counts.max()),
                )
                ax.plot(
                    [
                        np.deg2rad(mean_direction),
                        np.deg2rad(mean_direction),
                    ],
                    [
                        0.0,
                        vtest["R"] * local_rmax,
                    ],
                    linewidth=3,
                    color=color,
                    alpha=0.95,
                )

        if rmax is not None:
            ax.set_ylim(0, rmax)
        elif all_counts:
            global_rmax = max(
                float(np.max(counts))
                for counts in all_counts
            )
            ax.set_ylim(
                0,
                global_rmax * 1.15,
            )

        reference_direction = float(mu0_deg)
        if (
                fold_for_display
                and support_deg > 90.0
        ):
            reference_direction %= 180.0
            if reference_direction > 90.0:
                reference_direction = (
                        180.0 - reference_direction
                )

        reference_rad = np.deg2rad(
            reference_direction
        )
        ax.plot(
            [reference_rad, reference_rad],
            [0.0, ax.get_ylim()[1]],
            linewidth=4,
            color="black",
            alpha=0.20,
            label="Reference {:.0f}°".format(mu0_deg),
        )

        ax.set_title(
            "Relevant vs. irrelevant semantic-axis alignment",
            pad=12,
        )
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_thetamin(0)
        ax.set_thetamax(plot_support_deg)

        if plot_support_deg <= 90.0:
            tick_degrees = [0, 30, 60, 90]
        else:
            tick_degrees = [0, 45, 90, 135, 180]

        ax.set_xticks(
            np.deg2rad(tick_degrees)
        )
        ax.set_xticklabels(
            [
                "{}°".format(value)
                for value in tick_degrees
            ]
        )
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.35, 1.15),
            frameon=True,
        )

        if show_text:
            lines = []
            for title in titles:
                result = results[title]
                if result is None:
                    continue

                lines.append(
                    "{}: n={}, R={:.2f}, μ={:.1f}°, p={:.2e}".format(
                        title,
                        result["n"],
                        result["R"],
                        result["mu_deg"],
                        result["p"],
                    )
                )

            ax.text(
                0.5,
                -0.08,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                bbox={
                    "boxstyle": "round",
                    "fc": "white",
                    "ec": "0.7",
                    "alpha": 0.9,
                },
            )

        return results

    def _angle_deg(
            self,
            vector_a,
            vector_b,
            unsigned=False,
    ):
        """Return the angle between two vectors in degrees."""
        vector_a = self._unit(vector_a)
        vector_b = self._unit(vector_b)

        dot_product = float(
            np.dot(
                vector_a,
                vector_b,
            )
        )
        if unsigned:
            dot_product = abs(dot_product)

        dot_product = np.clip(
            dot_product,
            -1.0,
            1.0,
        )
        return np.degrees(
            np.arccos(dot_product)
        )

    def _collect_pair_angles(
            self,
            semantic_axes_raw,
            unsigned=False,
    ):
        """Collect pairwise task/emotion/gender angles across subjects."""
        pair_angles = {
            "Task-Gender": [],
            "Task-Emotion": [],
            "Gender-Emotion": [],
        }

        for axes in semantic_axes_raw.values():
            if not all(
                    key in axes
                    for key in (
                            "task",
                            "emotion",
                            "gender",
                    )
            ):
                continue

            pair_angles["Task-Gender"].append(
                self._angle_deg(
                    axes["task"],
                    axes["gender"],
                    unsigned=unsigned,
                )
            )
            pair_angles["Task-Emotion"].append(
                self._angle_deg(
                    axes["task"],
                    axes["emotion"],
                    unsigned=unsigned,
                )
            )
            pair_angles["Gender-Emotion"].append(
                self._angle_deg(
                    axes["gender"],
                    axes["emotion"],
                    unsigned=unsigned,
                )
            )

        return pair_angles

    @staticmethod
    def _circ_mean_and_R(angles_rad):
        """Return circular mean direction and mean resultant length."""
        mean_cos = np.mean(
            np.cos(angles_rad)
        )
        mean_sin = np.mean(
            np.sin(angles_rad)
        )
        mean_direction = np.arctan2(
            mean_sin,
            mean_cos,
        )
        resultant_length = np.sqrt(
            mean_cos * mean_cos
            + mean_sin * mean_sin
        )
        return mean_direction, resultant_length

    def _v_test_perm(
            self,
            angles_deg,
            mu0_deg=90.0,
            n_perm=20000,
            seed=0,
            support_deg=180.0,
    ):
        """Run a permutation-based directional V-test."""
        rng = np.random.RandomState(seed)

        angles = np.asarray(
            angles_deg,
            dtype=float,
        )
        angles = angles[np.isfinite(angles)]

        if len(angles) == 0:
            return {
                "n": 0,
                "mu_deg": np.nan,
                "R": np.nan,
                "V": np.nan,
                "p": np.nan,
            }

        scale = 360.0 / float(support_deg)
        mapped_angles = np.deg2rad(
            angles * scale
        )
        mapped_reference = np.deg2rad(
            mu0_deg * scale
        )

        mean_direction, resultant_length = (
            self._circ_mean_and_R(mapped_angles)
        )
        observed_v = (
                resultant_length
                * np.cos(
            mean_direction - mapped_reference
        )
        )

        permuted_v = np.zeros(
            n_perm,
            dtype=float,
        )

        for i in range(n_perm):
            null_angles = rng.uniform(
                0.0,
                support_deg,
                size=len(angles),
            )
            mapped_null = np.deg2rad(
                null_angles * scale
            )
            null_mean, null_resultant = (
                self._circ_mean_and_R(
                    mapped_null
                )
            )
            permuted_v[i] = (
                    null_resultant
                    * np.cos(
                null_mean - mapped_reference
            )
            )

        p_value = (
                          np.sum(permuted_v >= observed_v) + 1.0
                  ) / (n_perm + 1.0)

        mean_deg = (
                           np.rad2deg(mean_direction) % 360.0
                   ) * (support_deg / 360.0)

        return {
            "n": int(len(angles)),
            "mu_deg": float(mean_deg),
            "R": float(resultant_length),
            "V": float(observed_v),
            "p": float(p_value),
            "alpha_one_sided": float(
                self.alpha_one_sided
            ),
            "significant": bool(
                p_value < self.alpha_one_sided
            ),
        }

    def plot_orthogonality_rose_combined(
            self,
            ax,
            semantic_axes_raw,
            unsigned=False,
            mu0_deg=90.0,
            bins=18,
            n_perm=20000,
            seed=0,
            fold_for_display=False,
            rmax=None,
            alpha=0.35,
            histtype="bar",
            show_mean_lines=True,
            show_text=True,
            colors=None,
    ):
        """Plot the three task-relevant pairwise angle distributions."""
        pair_angles = self._collect_pair_angles(
            semantic_axes_raw,
            unsigned=unsigned,
        )

        support_deg = 90.0 if unsigned else 180.0
        titles = [
            "Task-Gender",
            "Task-Emotion",
            "Gender-Emotion",
        ]

        if colors is None:
            colors = {
                "Task-Gender": Constants.COLOR_G2G,
                "Task-Emotion": Constants.COLOR_E2E,
                "Gender-Emotion": Constants.COLOR_TASK_ARROW,
            }

        if fold_for_display and support_deg > 90.0:
            plot_support_deg = 90.0
        else:
            plot_support_deg = float(support_deg)

        edges = np.linspace(
            0.0,
            np.deg2rad(plot_support_deg),
            bins + 1,
        )
        widths = np.diff(edges)
        centers = edges[:-1] + widths / 2.0

        results = {}
        all_counts = []

        for i, title in enumerate(titles):
            angles = np.asarray(
                pair_angles[title],
                dtype=float,
            )
            angles = angles[np.isfinite(angles)]

            if len(angles) == 0:
                results[title] = None
                continue

            vtest = self._v_test_perm(
                angles_deg=angles,
                mu0_deg=mu0_deg,
                n_perm=n_perm,
                seed=seed + i,
                support_deg=support_deg,
            )
            results[title] = vtest

            plot_angles = angles.copy()
            if fold_for_display and support_deg > 90.0:
                folded = np.mod(
                    plot_angles,
                    support_deg,
                )
                plot_angles = np.where(
                    folded > 90.0,
                    180.0 - folded,
                    folded,
                )

            counts, _ = np.histogram(
                np.deg2rad(plot_angles),
                bins=edges,
            )
            all_counts.append(counts)

            if histtype == "step":
                theta_step = np.repeat(
                    edges,
                    2,
                )[1:-1]
                radius_step = np.repeat(
                    counts,
                    2,
                )
                ax.plot(
                    theta_step,
                    radius_step,
                    linewidth=2.5,
                    color=colors[title],
                    label=title,
                )
            else:
                ax.bar(
                    centers,
                    counts,
                    width=widths,
                    bottom=0.0,
                    align="center",
                    color=colors[title],
                    alpha=alpha,
                    edgecolor="black",
                    linewidth=0.5,
                    label=title,
                )

            if show_mean_lines:
                mean_direction = vtest["mu_deg"]
                if (
                        fold_for_display
                        and support_deg > 90.0
                ):
                    mean_direction %= 180.0
                    if mean_direction > 90.0:
                        mean_direction = (
                                180.0 - mean_direction
                        )

                local_rmax = max(
                    1.0,
                    float(counts.max()),
                )
                mean_rad = np.deg2rad(
                    mean_direction
                )
                ax.plot(
                    [mean_rad, mean_rad],
                    [
                        0.0,
                        vtest["R"] * local_rmax,
                    ],
                    linewidth=3,
                    color=colors[title],
                    alpha=0.95,
                )

        if rmax is not None:
            ax.set_ylim(0, rmax)
        elif all_counts:
            global_rmax = max(
                float(np.max(counts))
                for counts in all_counts
            )
            ax.set_ylim(
                0,
                global_rmax * 1.15,
            )

        reference_direction = float(mu0_deg)
        if (
                fold_for_display
                and support_deg > 90.0
        ):
            reference_direction %= 180.0
            if reference_direction > 90.0:
                reference_direction = (
                        180.0 - reference_direction
                )

        reference_rad = np.deg2rad(
            reference_direction
        )
        ax.plot(
            [reference_rad, reference_rad],
            [0.0, ax.get_ylim()[1]],
            linewidth=4,
            color="black",
            alpha=0.20,
            label="Reference {:.0f}°".format(mu0_deg),
        )

        ax.set_title(
            "Semantic-axis orthogonality",
            pad=12,
        )
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_thetamin(0)
        ax.set_thetamax(plot_support_deg)

        if plot_support_deg <= 90.0:
            tick_degrees = [0, 30, 60, 90]
        else:
            tick_degrees = [0, 45, 90, 135, 180]

        ax.set_xticks(
            np.deg2rad(tick_degrees)
        )
        ax.set_xticklabels(
            [
                "{}°".format(value)
                for value in tick_degrees
            ]
        )
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.35, 1.15),
            frameon=True,
        )

        if show_text:
            lines = []
            for title in titles:
                result = results[title]
                if result is None:
                    continue

                lines.append(
                    "{}: n={}, R={:.2f}, μ={:.1f}°, p={:.2e}".format(
                        title,
                        result["n"],
                        result["R"],
                        result["mu_deg"],
                        result["p"],
                    )
                )

            ax.text(
                0.5,
                -0.08,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                bbox={
                    "boxstyle": "round",
                    "fc": "white",
                    "ec": "0.7",
                    "alpha": 0.9,
                },
            )

        return results
