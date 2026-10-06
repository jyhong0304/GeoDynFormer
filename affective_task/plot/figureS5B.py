"""Generate Extended Data Figure 5b: exemplar trajectories across stimuli."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from affective_task.utils import Constants, save_figure
from affective_task.visualization import PlotModelLatents


class FigureS5B:
    """Plot four exemplar repeat/switch trajectory panels."""

    analysis_dir = "model_analysis"
    stats_fn = "holdout_outputs_01SD.pkl"

    project_root = Path(__file__).resolve().parents[2]
    patient_rs_path = project_root / "dataset" / "patient_rs.pkl"

    figsize = (10, 10)
    figdpi = 300
    T_POST = 1600

    PANEL_SPECS = [
        {
            "subject_id": "63",
            "stimulus": "male_happy",
            "grid": (slice(0, 7), slice(0, 7)),
            "elev": 20,
            "azim": 105,
            "xlim": [-10, 20],
            "ylim": [-12, 0],
            "zlim": [-12.5, 2.5],
            "plot_t_posts": {
                "g2g": 1500,
                "e2g": 1600,
                "e2e": 1500,
                "g2e": 2000,
            },
        },
        {
            "subject_id": "25",
            "stimulus": "male_angry",
            "grid": (slice(0, 7), slice(8, 15)),
            "elev": 30,
            "azim": 130,
            "xlim": [-10, 22],
            "ylim": [-4, 11],
            "zlim": [-4, 14],
            "plot_t_posts": {
                "g2g": 1500,
                "e2g": 1500,
                "e2e": 1500,
                "g2e": 1600,
            },
        },
        {
            "subject_id": "34",
            "stimulus": "female_happy",
            "grid": (slice(8, 15), slice(0, 7)),
            "elev": 40,
            "azim": 60,
            "xlim": [2.5, 17.5],
            "ylim": [-20, 5],
            "zlim": [-15, 15],
            "plot_t_posts": {
                "g2g": 1500,
                "e2g": 2000,
                "e2e": 1500,
                "g2e": 1600,
            },
        },
        {
            "subject_id": "44",
            "stimulus": "female_angry",
            "grid": (slice(8, 15), slice(8, 15)),
            "elev": 70,
            "azim": 110,
            "xlim": [-14, 2.5],
            "ylim": [-12.5, 17.5],
            "zlim": [-17.5, 12.5],
            "plot_t_posts": {
                "g2g": 1400,
                "e2g": 1600,
                "e2e": 1000,
                "g2e": 1600,
            },
        },
    ]

    def __init__(self, model_dir, save_dir, patients_id, rand_seed, n_boot):
        self.model_dir = model_dir
        self.save_dir = save_dir

        # Retain the common figure-class constructor signature.
        _ = patients_id, rand_seed, n_boot

        self.subject_stats = {}
        self.group_rotations = None

    def make_figure(self):
        """Generate, save, and return Extended Data Figure 5b."""
        print("Making Figure S5b...")
        self._run_preprocessing()

        print("Stats for Figure S5b")
        print("--------------------")

        fig = self._plot_figure_get_stats()

        save_figure(
            fig,
            self.save_dir,
            "FigS5b",
        )
        return fig

    def _run_preprocessing(self):
        """Load only the four exemplar models and their rotation matrices."""
        for spec in self.PANEL_SPECS:
            subject_id = spec["subject_id"]
            if subject_id in self.subject_stats:
                continue

            stats_path = (
                    Path(self.model_dir)
                    / "s{}".format(subject_id)
                    / self.analysis_dir
                    / self.stats_fn
            )
            with stats_path.open("rb") as file:
                self.subject_stats[subject_id] = pickle.load(file)

        with self.patient_rs_path.open("rb") as file:
            self.group_rotations = pickle.load(file)

        print(
            "Data successfully loaded from {}".format(
                self.patient_rs_path
            )
        )

    def _plot_figure_get_stats(self):
        """Assemble the four exemplar 3D trajectory panels."""
        fig = plt.figure(
            constrained_layout=False,
            figsize=self.figsize,
            dpi=self.figdpi,
        )
        gs = fig.add_gridspec(15, 15)

        for spec in self.PANEL_SPECS:
            rows, cols = spec["grid"]
            ax = fig.add_subplot(
                gs[rows, cols],
                projection="3d",
            )

            plot_kwargs = {
                "xlim": spec["xlim"],
                "ylim": spec["ylim"],
                "zlim": spec["zlim"],
                "annotate": "global",
                "R": self.group_rotations[
                    spec["subject_id"]
                ],
            }

            self._make_stimulus_panel(
                ax=ax,
                stats=self.subject_stats[
                    spec["subject_id"]
                ],
                stimulus=spec["stimulus"],
                plot_t_posts=spec["plot_t_posts"],
                elev=spec["elev"],
                azim=spec["azim"],
                plot_kwargs=plot_kwargs,
            )

        return fig

    def _make_stimulus_panel(
            self,
            ax,
            stats,
            stimulus,
            plot_t_posts,
            elev,
            azim,
            plot_kwargs,
            t_post=T_POST,
    ):
        """Plot repeat/switch trajectories for one stimulus configuration."""
        plotter = PlotModelLatents(
            stats,
            post_on_dur=t_post,
            plot_pre_onset=False,
        )

        trajectory_specs = [
            ("g2g", Constants.COLOR_G2G, "-"),
            ("e2g", Constants.COLOR_G2G, "--"),
            ("e2e", Constants.COLOR_E2E, "-"),
            ("g2e", Constants.COLOR_E2E, "--"),
        ]

        for transition, color, line_style in trajectory_specs:
            plot_kwargs["plot_t_posts"] = [
                plot_t_posts[transition]
            ]
            plot_kwargs["line_styles"] = [
                line_style
            ]
            plot_kwargs["colors"] = [
                color
            ]

            plot_method = getattr(
                plotter,
                "plot_{}_{}".format(
                    transition,
                    stimulus,
                ),
            )
            plot_method(
                ax,
                elev=elev,
                azim=azim,
                plot_task_centroid=False,
                **plot_kwargs
            )
