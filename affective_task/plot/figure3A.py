"""Generate Figure 3a: subject-specific and aligned latent geometry."""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from affective_task.utils import (
    Constants,
    compute_aligned_projections_joint,
    overlay_semantic_axes_corner,
    save_figure,
)
from affective_task.visualization import PlotModelLatents


class Figure3A:
    """Plot local and group-aligned repeat-trial latent geometry."""

    analysis_dir = "model_analysis"
    stats_fn = "holdout_outputs_01SD.pkl"
    fp_fn = "fixed_points.pkl"

    # Rotation matrices are reused by later figure-generation code.
    patient_rs = "../../dataset/patient_rs.pkl"

    exemplar_id = "32"
    figsize = (10, 5)
    figdpi = 300
    T_POST = 1500

    def __init__(self, model_dir, save_dir, patients_id, rand_seed, n_boot):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.patients_id = patients_id

        # Retain the common figure-class constructor signature. These two
        # arguments are not required by the active Figure 3a analysis.
        _ = rand_seed
        _ = n_boot

        self.t_start_ind = 5
        self.group_fp = {}

    def make_figure(self):
        """Generate and save Figure 3a."""
        print("Making Figure 3a...")
        self._run_preprocessing()

        print("Stats for Figure 3a")
        print("------------------")

        fig = self._plot_figure_get_stats()
        save_figure(fig, self.save_dir, "Fig3a")
        return fig

    def _run_preprocessing(self):
        """Load subject outputs, align latent geometry, and save rotations."""
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

            if patient_id == self.exemplar_id:
                self.exemplar_stats = expt_stats

            fp_path = os.path.join(
                subject_dir,
                self.analysis_dir,
                self.fp_fn,
            )
            with open(fp_path, "rb") as path:
                fixed_points = pickle.load(path)

            self.group_fp[str(patient_id)] = fixed_points

        # Construct the subject-level representation used to derive the
        # cross-subject alignment. This preserves the original analysis.
        rows = []
        for patient_id, fixed_points in self.group_fp.items():
            latents = expt_stats.windowed["umap_latents"][:, :, :]
            n_pre = self.t_start_ind

            for task_cue in [0, 1]:
                for stim_gender in [0, 1]:
                    for stim_emotion in [0, 1]:
                        zloc_umap = fixed_points.loc[
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

                        x = np.mean(latents[:, filtered_inds, 0], axis=1)
                        y = np.mean(latents[:, filtered_inds, 1], axis=1)
                        z = np.mean(latents[:, filtered_inds, 2], axis=1)

                        task = "gender" if task_cue == 0 else "emotion"
                        gender = "male" if stim_gender == 0 else "female"
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
                                x[n_pre],
                                y[n_pre],
                                z[n_pre],
                                zloc_umap[0],
                                zloc_umap[1],
                                zloc_umap[2],
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
            group_diagnostics,
            self.group_Rs,
            _,
        ) = compute_aligned_projections_joint(group_representation)

        print(group_diagnostics)

        # Preserve the original rotation matrices for downstream figures.
        if os.path.exists(self.patient_rs):
            print("Patient rotation matrices already exist")
        else:
            with open(self.patient_rs, "wb") as path:
                pickle.dump(self.group_Rs, path, protocol=4)
            print("Patient rotation matrices saved")

    def _plot_figure_get_stats(self):
        """Assemble the local-space and aligned-space panels."""
        fig = plt.figure(
            constrained_layout=False,
            figsize=self.figsize,
            dpi=self.figdpi,
        )
        gs = fig.add_gridspec(8, 17)

        subject_rotation = self.group_Rs[self.exemplar_id]
        subject_fixed_points = self.group_fp[self.exemplar_id]

        # Left: exemplar trajectories in the subject-specific UMAP space.
        local_plot_kwargs = {
            "xlim": [-5, 15],
            "ylim": [-5, 15],
            "zlim": [-5, 15],
            "annotate": "local",
            "colors": [
                Constants.COLOR_G2G,
                Constants.COLOR_G2G,
                Constants.COLOR_E2E,
                Constants.COLOR_E2E,
            ],
            "plot_t_posts": [1200, 1200, 1200, 1200],
        }

        local_ax = fig.add_subplot(
            gs[0:8, 0:8],
            projection="3d",
        )
        self._make_panel_B(
            local_ax,
            is_full_stimulus=False,
            p_kwargs=local_plot_kwargs,
            elev=25,
            azim=25,
            fixed_points=subject_fixed_points,
        )
        overlay_semantic_axes_corner(
            local_ax,
            subject_rotation,
            corner=(0.90, 0.10, 0.90),
            scale=0.18,
            text_offset=1.1,
            alpha=0.8,
        )

        # Right: the same trajectories after projection to the common axes.
        global_plot_kwargs = {
            "annotate": "global",
            "colors": [
                Constants.COLOR_G2G,
                Constants.COLOR_G2G,
                Constants.COLOR_E2E,
                Constants.COLOR_E2E,
            ],
            "plot_t_posts": [1200, 1200, 1200, 1200],
            "R": subject_rotation,
        }

        global_ax = fig.add_subplot(
            gs[0:8, 9:17],
            projection="3d",
        )
        self._make_panel_B(
            global_ax,
            is_full_stimulus=False,
            p_kwargs=global_plot_kwargs,
            elev=25,
            azim=25,
            fixed_points=subject_fixed_points,
        )
        overlay_semantic_axes_corner(
            global_ax,
            np.eye(3),
            corner=(0.90, 0.10, 0.90),
            scale=0.18,
            text_offset=1.1,
            alpha=0.8,
        )

        return fig

    def _make_panel_B(
            self,
            ax,
            elev=30,
            azim=60,
            p_kwargs=None,
            t_post=T_POST,
            is_full_stimulus=True,
            is_switch=False,
            fixed_points=None,
    ):
        """Plot exemplar latent trajectories in a 3D axis."""
        if p_kwargs is None:
            p_kwargs = {
                "xlim": [-20, 20],
                "ylim": [-10, 8],
                "zlim": [-10, 10],
            }

        plotter = PlotModelLatents(
            self.exemplar_stats,
            post_on_dur=t_post,
            plot_pre_onset=False,
            fixed_points=fixed_points,
        )

        if is_full_stimulus:
            plotter.plot_full_conditions(
                ax,
                elev=elev,
                azim=azim,
                plot_task_centroid=False,
                is_switch=is_switch,
                **p_kwargs
            )
        else:
            plotter.plot_related_stim_conditions(
                ax,
                elev=elev,
                azim=azim,
                plot_task_centroid=False,
                is_switch=is_switch,
                **p_kwargs
            )
