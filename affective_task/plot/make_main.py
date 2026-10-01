"""Generate all main manuscript figures."""

import argparse
import os
import time

import matplotlib

# Select the interactive backend before importing pyplot.
# matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from preprocessing import Preprocess
from figure2 import Figure2
from figure3A import Figure3A
from figure3BCDE import Figure3BCDE
from figure4 import Figure4
from figure5 import Figure5
from figure6A import Figure6A
from figure6BCDE import Figure6BCDE
from figure6FG import Figure6FG
from figure7B import Figure7B
from figure7CDEFGH import Figure7CDEFGH
from figure8AE import Figure8AE
from figure8BF import Figure8BF
from figure8CD import Figure8CD
from figure8GH import Figure8GH

# ----------------------------------------------------------------------
# Global plotting / analysis configuration
# ----------------------------------------------------------------------
BATCH_SIZE = 512
RAND_SEED = 1
N_BOOT = 1000
FONT_SIZE = 5

DEFAULT_PATIENT_IDS = [
    "20",
    "21",
    "24",
    "25",
    "26",
    "27",
    "29",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "39",
    "44",
    "57",
    "63",
    "66_1",
    "68_2",
    "69_2",
    "71_2",
]


def _configure_matplotlib():
    """Apply manuscript-wide Matplotlib settings."""
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.labelsize"] = FONT_SIZE
    plt.rcParams["axes.titlesize"] = FONT_SIZE
    plt.rcParams["xtick.labelsize"] = FONT_SIZE
    plt.rcParams["ytick.labelsize"] = FONT_SIZE
    plt.rcParams["legend.fontsize"] = FONT_SIZE


def _build_parser():
    """Create the command-line parser for main-figure generation."""
    parser = argparse.ArgumentParser(
        description="Generate all main manuscript figures."
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        help=(
            "Directory containing model subdirectories "
            "(for example, ../trained_models)."
        ),
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        help=(
            "Optional comma-separated participant IDs. "
            "If omitted, the manuscript participant set is used."
        ),
    )
    parser.add_argument(
        "-f",
        "--figure_dir",
        default="figures",
        nargs="?",
        type=str,
        help="Directory for generated figures (default: figures).",
    )
    parser.add_argument(
        "-p",
        "--do-preprocessing",
        action="store_true",
        help="Recompute model-analysis outputs before plotting.",
    )

    return parser


def _get_patient_ids(patient_id_arg):
    """Parse a comma-separated participant list or return the default set."""
    if patient_id_arg is None:
        return list(DEFAULT_PATIENT_IDS)

    return [
        patient_id.strip()
        for patient_id in patient_id_arg.split(",")
        if patient_id.strip()
    ]


def _run_preprocessing_if_requested(
        model_dir,
        patient_ids,
        do_preprocessing,
):
    """Recompute model-analysis outputs when explicitly requested."""
    if not do_preprocessing:
        return

    preprocessing = Preprocess(
        model_dir,
        patient_ids,
        RAND_SEED,
        batch_size=BATCH_SIZE,
    )
    preprocessing.run_preprocessing()


def _make_main_figures(
        model_dir,
        figure_dir,
        patient_ids,
):
    """Generate Figures 2-8 in manuscript order."""

    # ------------------------------------------------------------------
    # Figure 2
    #   a: exemplar RT distributions
    #   b: participant/model mean-RT relationship
    #   c-d: participant/model switch costs
    # ------------------------------------------------------------------
    figure2 = Figure2(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure2.make_figure()

    # ------------------------------------------------------------------
    # Figure 3
    #   a: repeat-trial latent trajectories
    #   b-c: task-relevant / task-irrelevant semantic-axis geometry
    #   d-e: Gram-matrix orthogonality analysis
    # ------------------------------------------------------------------
    figure3a = Figure3A(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure3a.make_figure()

    figure3bcde = Figure3BCDE(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure3bcde.make_figure()

    # ------------------------------------------------------------------
    # Figure 4
    #   Repeat and switch trajectories showing onset-state separation.
    # ------------------------------------------------------------------
    figure4 = Figure4(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure4.make_figure()

    # ------------------------------------------------------------------
    # Figure 5
    #   Latent trajectories, LDA analyses, and fixed-point distances.
    # ------------------------------------------------------------------
    figure5 = Figure5(
        model_dir,
        figure_dir,
        patient_ids,
    )
    figure5.make_figure()

    # ------------------------------------------------------------------
    # Figure 6
    #   a: exemplar onset-state geometry
    #   b-c: onset-distance / switch-cost correlations
    #   d: E2G/G2E onset-distance comparison
    #   e: repeated-measures correlation
    #   f-g: split-half reliability
    # ------------------------------------------------------------------
    figure6a = Figure6A(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure6a.make_figure()

    figure6bcde = Figure6BCDE(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure6bcde.make_figure()

    figure6fg = Figure6FG(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure6fg.make_figure()

    # ------------------------------------------------------------------
    # Figure 7
    #   b: exemplar t_c-based trajectory decomposition
    #   c-h: trajectory-geometry / HDDM analyses
    # ------------------------------------------------------------------
    figure7b = Figure7B(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure7b.make_figure()

    figure7cdefgh = Figure7CDEFGH(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure7cdefgh.make_figure()

    # ------------------------------------------------------------------
    # Figure 8
    #   a,e: exemplar aligned task trajectories
    #   b,f: A_rel / A_irr transition comparisons
    #   c,d: early A_irr vs. HDDM non-decision time
    #   g,h: late A_rel vs. HDDM drift rate
    # ------------------------------------------------------------------
    figure8ae = Figure8AE(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure8ae.make_figure()

    figure8bf = Figure8BF(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure8bf.make_figure()

    figure8cd = Figure8CD(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure8cd.make_figure()

    figure8gh = Figure8GH(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure8gh.make_figure()


def main():
    """Entry point for main-figure generation."""
    args = _build_parser().parse_args()

    model_dir = args.model_dir
    figure_dir = args.figure_dir
    patient_ids = _get_patient_ids(
        args.patient_id
    )

    os.makedirs(
        figure_dir,
        exist_ok=True,
    )

    _configure_matplotlib()

    print(
        "Number of patients: {}\n"
        "Patient ID List: {}".format(
            len(patient_ids),
            patient_ids,
        )
    )

    start_time = time.time()

    _run_preprocessing_if_requested(
        model_dir=model_dir,
        patient_ids=patient_ids,
        do_preprocessing=args.do_preprocessing,
    )

    _make_main_figures(
        model_dir=model_dir,
        figure_dir=figure_dir,
        patient_ids=patient_ids,
    )

    plt.show()

    run_time = time.time() - start_time
    print(
        "Run time: {:.2f}s".format(
            run_time
        )
    )


if __name__ == "__main__":
    main()
