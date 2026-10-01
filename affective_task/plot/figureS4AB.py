"""Generate Extended Data Figure 4a-b: latent geometry from multiple views."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from affective_task.utils import (
    Constants,
    overlay_semantic_axes_corner,
    save_figure,
)
from affective_task.visualization import PlotModelLatents


class FigureS4AB:
    """Plot subject-25 related-stimulus latent trajectories from four viewpoints."""

    analysis_dir = "model_analysis"
    stats_fn = "holdout_outputs_01SD.pkl"
    fixed_points_fn = "fixed_points.pkl"

    project_root = Path(__file__).resolve().parents[2]
    patient_rs_path = project_root / "dataset" / "patient_rs.pkl"

    exemplar_id = "25"

    figsize = (9, 5.5)
    figdpi = 300
    T_POST = 1500

    TRAJECTORY_COLORS = [
        Constants.COLOR_G2G,
        Constants.COLOR_G2G,
        Constants.COLOR_E2E,
        Constants.COLOR_E2E,
    ]

    VIEW_SPECS = [
        {
            "grid": (slice(0, 6), slice(0, 6)),
            "xlim": [-5, 15],
            "ylim": [-5, 15],
            "zlim": [-5, 15],
            "elev": -10,
            "azim": 20,
        },
        {
            "grid": (slice(7, 13), slice(0, 6)),
            "xlim": [-6, 16],
            "ylim": [-6, 16],
            "zlim": [-6, 16],
            "elev": 90,
            "azim": -90,
        },
        {
            "grid": (slice(7, 13), slice(7, 13)),
            "xlim": [-6, 16],
            "ylim": [-6, 16],
            "zlim": [-6, 16],
            "elev": 0,
            "azim": -90,
        },
        {
            "grid": (slice(0, 6), slice(7, 13)),
            "xlim": [-6, 16],
            "ylim": [-6, 16],
            "zlim": [-6, 16],
            "elev": 0,
            "azim": 0,
        },
    ]

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

        # Retain the common figure-class constructor signature.
        _ = patients_id, rand_seed, n_boot

        self.expt_stats = None
        self.fixed_points = None
        self.rotation = None

    def make_figure(self):
        """Generate, save, and return Extended Data Figure 4a-b."""
        print("Making Figure S4ab...")
        self._run_preprocessing()

        print("Stats for Figure S4ab")
        print("---------------------")

        fig = self._plot_figure_get_stats()

        save_figure(
            fig,
            self.save_dir,
            "FigS4ab",
        )
        return fig

    def _run_preprocessing(self):
        """Load the exemplar subject data, fixed points, and semantic rotation."""
        subject_dir = (
                Path(self.model_dir)
                / "s{}".format(self.exemplar_id)
                / self.analysis_dir
        )

        stats_path = (
                subject_dir
                / self.stats_fn
        )
        fixed_points_path = (
                subject_dir
                / self.fixed_points_fn
        )

        with stats_path.open("rb") as file:
            self.expt_stats = pickle.load(file)

        with fixed_points_path.open("rb") as file:
            self.fixed_points = pickle.load(file)

        with self.patient_rs_path.open("rb") as file:
            group_rotations = pickle.load(file)

        self.rotation = group_rotations[
            self.exemplar_id
        ]

        print(
            "Data successfully loaded from {}".format(
                self.patient_rs_path
            )
        )

    def _plot_figure_get_stats(self):
        """Render the four views of the same related-stimulus geometry."""
        fig = plt.figure(
            constrained_layout=False,
            figsize=self.figsize,
            dpi=self.figdpi,
        )
        gs = fig.add_gridspec(
            13,
            13,
        )

        for spec in self.VIEW_SPECS:
            row_slice, col_slice = spec[
                "grid"
            ]

            ax = fig.add_subplot(
                gs[
                    row_slice,
                    col_slice,
                ],
                projection="3d",
            )

            plot_kwargs = {
                "xlim": spec["xlim"],
                "ylim": spec["ylim"],
                "zlim": spec["zlim"],
                "colors": self.TRAJECTORY_COLORS,
                "plot_t_posts": [
                    1200,
                    1200,
                    1200,
                    1200,
                ],
            }

            self._make_related_stimulus_panel(
                ax=ax,
                elev=spec["elev"],
                azim=spec["azim"],
                plot_kwargs=plot_kwargs,
            )

            overlay_semantic_axes_corner(
                ax,
                self.rotation,
                corner=(0.90, 0.10, 0.90),
                scale=0.18,
                text_offset=1.1,
                alpha=0.8,
            )

        return fig

    def _make_related_stimulus_panel(
            self,
            ax,
            elev,
            azim,
            plot_kwargs,
    ):
        """Plot repeat-trial trajectories for task-related stimulus conditions."""
        plotter = PlotModelLatents(
            self.expt_stats,
            post_on_dur=self.T_POST,
            plot_pre_onset=False,
            fixed_points=self.fixed_points,
        )

        plotter.plot_related_stim_conditions(
            ax,
            elev=elev,
            azim=azim,
            plot_task_centroid=False,
            is_switch=False,
            **plot_kwargs
        )
