"""Weights & Biases logging utilities for GeoDynFormer training."""

import wandb


class WanDBLogger:
    """Logger for tracking GeoDynFormer experiments with Weights & Biases."""

    def __init__(self, expt_name, project_name="GeoDynFormer", mode="online"):
        """
        Initialize a Weights & Biases run.

        Parameters
        ----------
        expt_name : str
            Name of the experiment/run.
        project_name : str, optional
            Name of the Weights & Biases project.
        mode : str, optional
            Weights & Biases logging mode, e.g. "online", "offline", or
            "disabled".
        """
        self.expt_name = expt_name
        self.project_name = project_name
        self.mode = mode

        wandb.init(
            project=self.project_name,
            name=self.expt_name,
            mode=self.mode,
        )

    def log_metrics(
            self,
            metrics,
            name,
            epoch,
            epoch_end=False,
            iteration=None,
            anneal_param=None,
    ):
        """
        Log training or validation metrics.

        Parameters
        ----------
        metrics : dict
            Dictionary containing metric names and values.
        name : str
            Prefix used to group metrics, e.g. "train" or "validation".
        epoch : int
            Current training epoch.
        epoch_end : bool, optional
            Whether the current logging call corresponds to the end of an
            epoch.
        iteration : int, optional
            Current training iteration.
        anneal_param : float, optional
            Current value of the annealing parameter.
        """
        log_data = {
            "{}/{}".format(name, key): value
            for key, value in metrics.items()
        }
        log_data["epoch"] = epoch

        if epoch_end:
            if iteration is not None:
                log_data["iteration"] = iteration

            if anneal_param is not None:
                log_data["anneal_param"] = anneal_param

        wandb.log(log_data)

    def log_sample_output(self, fig, epoch):
        """
        Log a model-output figure.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to upload to Weights & Biases.
        epoch : int
            Epoch associated with the figure.
        """
        key = "model_outputs/epoch{}".format(epoch)

        wandb.log(
            {
                key: wandb.Image(fig),
                "epoch": epoch,
            }
        )

    @staticmethod
    def finish():
        """Finish the active Weights & Biases run."""
        wandb.finish()
