"""Train a GeoDynFormer model on an affective task-switching dataset."""

import argparse
import time
from pathlib import Path

from affective_task_switching import Experiment

RAW_DATA_FILENAME = "data_pre_split.pkl"


def positive_int(value):
    """Parse a strictly positive integer from a command-line argument."""
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def nonnegative_int(value):
    """Parse a non-negative integer from a command-line argument."""
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return value


def parse_args():
    """Parse command-line arguments for GeoDynFormer training."""
    parser = argparse.ArgumentParser(
        description=(
            "Train GeoDynFormer using a preprocessed affective "
            "task-switching dataset."
        )
    )

    parser.add_argument(
        "--raw_data_dir",
        type=Path,
        required=True,
        help=(
            "Directory containing the preprocessed training data file "
            "'{}'.".format(RAW_DATA_FILENAME)
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help=(
            "Parent directory for training outputs. The current run is "
            "saved under SAVE_DIR/EXPT_NAME."
        ),
    )
    parser.add_argument(
        "--expt_name",
        type=str,
        required=True,
        help="Name of the training run/experiment.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional path to a YAML configuration file. If omitted, the "
            "default model configuration is used."
        ),
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda:0",
        help="Device used for training, e.g. 'cuda:0' or 'cpu' (default: cuda:0).",
    )
    parser.add_argument(
        "-wd",
        "--wandb",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Weights & Biases logging name retained for compatibility with "
            "the original training command. W&B is the only supported logger."
        ),
    )
    parser.add_argument(
        "-rs",
        "--rand_seed",
        type=nonnegative_int,
        default=None,
        help="Override the random seed defined in the configuration file.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=positive_int,
        default=None,
        help="Override the training batch size defined in the configuration file.",
    )
    parser.add_argument(
        "-um",
        "--upscale_mult",
        type=positive_int,
        default=None,
        help=(
            "Override the training-set augmentation multiplier defined in "
            "the configuration file."
        ),
    )
    parser.add_argument(
        "-me",
        "--max_epochs",
        type=positive_int,
        default=None,
        help=(
            "Override the maximum number of training epochs. Early stopping "
            "may terminate training sooner."
        ),
    )

    return parser.parse_args()


def build_experiment_kwargs(args, save_dir):
    """Build configuration overrides passed to Experiment."""
    kwargs = {
        "logger_type": "wandb",
    }

    # Retain the original --wandb command-line interface. The current
    # WanDBLogger defines the actual W&B project name in logging.py.
    if args.wandb is not None:
        kwargs["log_save_dir"] = str(save_dir / args.wandb)

    if args.rand_seed is not None:
        kwargs["rand_seed"] = args.rand_seed

    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size

    if args.upscale_mult is not None:
        kwargs["train"] = {"upscale_mult": args.upscale_mult}

    if args.max_epochs is not None:
        kwargs["num_epochs"] = args.max_epochs

    return kwargs


def main():
    """Create and train a GeoDynFormer experiment."""
    args = parse_args()

    raw_data_dir = args.raw_data_dir.expanduser().resolve()
    save_dir = args.save_dir.expanduser().resolve()

    if args.config is None:
        config_path = None
    else:
        config_path = args.config.expanduser().resolve()

    raw_data_path = raw_data_dir / RAW_DATA_FILENAME
    if not raw_data_path.is_file():
        raise FileNotFoundError(
            "Expected preprocessed data file was not found: {}".format(
                raw_data_path
            )
        )

    if config_path is not None and not config_path.is_file():
        raise FileNotFoundError(
            "Configuration file was not found: {}".format(config_path)
        )

    experiment_dir = save_dir / args.expt_name
    processed_data_dir = experiment_dir / "processed_data"

    experiment_kwargs = build_experiment_kwargs(args, save_dir)

    start_time = time.perf_counter()

    experiment = Experiment(
        base_dir=str(experiment_dir),
        raw_data_dir=str(raw_data_dir),
        raw_fn=RAW_DATA_FILENAME,
        expt_name=args.expt_name,
        config=str(config_path) if config_path is not None else None,
        device=args.device,
        processed_dir=str(processed_data_dir),
        **experiment_kwargs
    )
    experiment.run_training()

    elapsed_seconds = time.perf_counter() - start_time
    print("Experiment run time: {:.2f} s".format(elapsed_seconds))


if __name__ == "__main__":
    main()
