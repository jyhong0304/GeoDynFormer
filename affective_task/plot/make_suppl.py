"""Generate all Extended Data manuscript figures."""

import argparse
import os
import time

import matplotlib.pyplot as plt

import matplotlib

# Select the interactive backend before importing pyplot.
matplotlib.use("TkAgg")

from preprocessing import Preprocess
from figureS1 import FigureS1
from figureS2 import FigureS2
from figureS3 import FigureS3
from figureS4AB import FigureS4AB
from figureS5A import FigureS5A
from figureS5B import FigureS5B
from figureS6 import FigureS6
from figureS7 import FigureS7
from figureS8A import FigureS8A
from figureS8B import FigureS8B

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
    """Create the command-line parser for Extended Data figure generation."""
    parser = argparse.ArgumentParser(
        description="Generate all Extended Data manuscript figures."
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


def _make_supplementary_figures(
        model_dir,
        figure_dir,
        patient_ids,
):
    """Generate Extended Data Figures 1-8 in manuscript order."""

    # ------------------------------------------------------------------
    # Extended Data Figure 1
    #   GeoDynFormer vs. task-DyVA architecture comparison.
    # ------------------------------------------------------------------
    figure_s1 = FigureS1(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure_s1.make_figure()

    # ------------------------------------------------------------------
    # Extended Data Figure 2
    #   Mean RTs across pure, repeat, and switch trial types.
    # ------------------------------------------------------------------
    figure_s2 = FigureS2(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure_s2.make_figure()

    # ------------------------------------------------------------------
    # Extended Data Figure 3
    #   Participant/model behavioral relationships.
    # ------------------------------------------------------------------
    figure_s3 = FigureS3(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure_s3.make_figure()

    # ------------------------------------------------------------------
    # Extended Data Figure 4a-b
    #   Additional latent-state trajectory views.
    # ------------------------------------------------------------------
    figure_s4ab = FigureS4AB(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure_s4ab.make_figure()

    # ------------------------------------------------------------------
    # Extended Data Figure 5
    #   a-b: exemplar repeat/switch trajectory visualizations.
    # ------------------------------------------------------------------
    figure_s5a = FigureS5A(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure_s5a.make_figure()

    figure_s5b = FigureS5B(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure_s5b.make_figure()

    # ------------------------------------------------------------------
    # Extended Data Figure 6
    #   Mean distance-RT correlation histograms.
    # ------------------------------------------------------------------
    figure_s6 = FigureS6(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure_s6.make_figure()

    # ------------------------------------------------------------------
    # Extended Data Figure 7
    #   HDDM parameter estimates across task transitions.
    # ------------------------------------------------------------------
    figure_s7 = FigureS7(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure_s7.make_figure()

    # ------------------------------------------------------------------
    # Extended Data Figure 8
    #   a: drift rate vs. trajectory length
    #   b: non-decision time vs. trajectory length
    # ------------------------------------------------------------------
    figure_s8a = FigureS8A(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure_s8a.make_figure()

    figure_s8b = FigureS8B(
        model_dir,
        figure_dir,
        patient_ids,
        RAND_SEED,
        N_BOOT,
    )
    figure_s8b.make_figure()


def main():
    """Entry point for Extended Data figure generation."""
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

    _make_supplementary_figures(
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
