"""Generate Extended Data Figure 5a: exemplar repeat/switch trajectories."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from affective_task.utils import Constants, save_figure
from affective_task.visualization import PlotModelLatents


class FigureS5A:
    """Plot exemplar repeat and switch trajectories for one participant."""

    analysis_dir = "model_analysis"
    stats_fn = "holdout_outputs_01SD.pkl"

    project_root = Path(__file__).resolve().parents[2]
    patient_rs_path = project_root / "dataset" / "patient_rs.pkl"

    exemplar_id = "63"
    figsize = (5, 5)
    figdpi = 300
    T_POST = 1600

    def __init__(self, model_dir, save_dir, patients_id, rand_seed, n_boot):
        self.model_dir = model_dir
        self.save_dir = save_dir

        # Retain the common figure-class constructor signature.
        _ = patients_id, rand_seed, n_boot

        self.expt_stats = None
        self.rotation = None

    def make_figure(self):
        """Generate, save, and return Extended Data Figure 5a."""
        print("Making Figure S5a...")
        self._run_preprocessing()

        print("Stats for Figure S5a")
        print("--------------------")

        fig = self._plot_figure_get_stats()

        save_figure(
            fig,
            self.save_dir,
            "FigS5a",
            save_html=True,
        )
        return fig

    def _run_preprocessing(self):
        """Load the exemplar model outputs and subject-specific rotation."""
        analysis_path = (
                Path(self.model_dir)
                / "s{}".format(self.exemplar_id)
                / self.analysis_dir
                / self.stats_fn
        )

        with analysis_path.open("rb") as file:
            self.expt_stats = pickle.load(file)

        with self.patient_rs_path.open("rb") as file:
            group_rotations = pickle.load(file)

        self.rotation = group_rotations[self.exemplar_id]

        print(
            "Data successfully loaded from {}".format(
                self.patient_rs_path
            )
        )

    def _plot_figure_get_stats(self):
        """Assemble the exemplar 3D trajectory panel."""
        fig = plt.figure(
            constrained_layout=False,
            figsize=self.figsize,
            dpi=self.figdpi,
        )
        gs = fig.add_gridspec(7, 7)

        ax = fig.add_subplot(
            gs[0:7, 0:7],
            projection="3d",
        )

        plot_kwargs = {
            "xlim": [-2.5, 15],
            "ylim": [-18, 3],
            "zlim": [-15, 0],
            "annotate": "global",
            "R": self.rotation,
        }

        self._make_exemplar_panel(
            ax=ax,
            plot_kwargs=plot_kwargs,
            elev=30,
            azim=100,
        )

        return fig

    def _make_exemplar_panel(
            self,
            ax,
            plot_kwargs,
            elev=30,
            azim=60,
            t_post=T_POST,
    ):
        """Plot G2G/E2G and E2E/G2E trajectories for the exemplar subject."""
        plotter = PlotModelLatents(
            self.expt_stats,
            post_on_dur=t_post,
            plot_pre_onset=False,
        )

        # G2G vs. E2G for the female-happy stimulus.
        plot_kwargs["plot_t_posts"] = [1500]
        plot_kwargs["line_styles"] = ["-"]
        plot_kwargs["colors"] = [Constants.COLOR_G2G]
        plotter.plot_g2g_female_happy(
            ax,
            elev=elev,
            azim=azim,
            plot_task_centroid=False,
            **plot_kwargs
        )

        plot_kwargs["plot_t_posts"] = [1800]
        plot_kwargs["line_styles"] = ["--"]
        plot_kwargs["colors"] = [Constants.COLOR_G2G]
        plotter.plot_e2g_female_happy(
            ax,
            elev=elev,
            azim=azim,
            plot_task_centroid=False,
            **plot_kwargs
        )

        # E2E vs. G2E for the female-happy stimulus.
        plot_kwargs["plot_t_posts"] = [1000]
        plot_kwargs["line_styles"] = ["-"]
        plot_kwargs["colors"] = [Constants.COLOR_E2E]
        plotter.plot_e2e_female_happy(
            ax,
            elev=elev,
            azim=azim,
            plot_task_centroid=False,
            **plot_kwargs
        )

        plot_kwargs["plot_t_posts"] = [2100]
        plot_kwargs["line_styles"] = ["--"]
        plot_kwargs["colors"] = [Constants.COLOR_E2E]
        plotter.plot_g2e_female_happy(
            ax,
            elev=elev,
            azim=azim,
            plot_task_centroid=False,
            **plot_kwargs
        )
