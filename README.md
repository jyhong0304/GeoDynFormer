# GeoDynFormer

Official implementation of GeoDynFormer accompanying the manuscript:

[Dynamics and geometry of emotion and cognition: an interpretable model of individual human behavior](https://www.biorxiv.org/content/10.64898/2026.09.08.749516v1.abstract)

Jinyung Hong et al.

GeoDynFormer is a subject-specific latent dynamical modeling framework for affective task-switching (ATS) behavior. The
framework models trial-by-trial behavioral responses as temporally evolving latent dynamics and enables geometric
analysis of latent-state trajectories.

## Repository contents

This repository contains code for:

- preprocessing affective task-switching behavioral data;
- training and evaluating subject-specific GeoDynFormer models;
- generating model responses;
- extracting and visualizing latent-state trajectories;
- UMAP and semantic-axis analyses;
- fixed-point, distance, tortuosity, reliability, and angular-alignment analyses;
- hierarchical drift-diffusion modeling and associated statistical analyses; and
- generating the figures reported in the manuscript.

## System requirements

### Software

The manuscript analyses used the following software:

- Python **3.7**
- PyTorch **1.9**
- HDDM **1.0.1**
- umap-learn **0.5.3**
- Pingouin **0.5.3**

All Python dependencies and their versions are listed in [`requirements.txt`](requirements.txt).

### Operating system

The code was tested on:

- **Ubuntu 24.04**

### Hardware

GeoDynFormer models used in the manuscript were trained on an **NVIDIA RTX Ada 6000 GPU**.

Users who wish to use GPU acceleration should ensure that their CUDA and NVIDIA driver configuration is compatible with
the installed PyTorch version. Runtime will depend on the available hardware.

## Installation

Clone the repository:

```bash
git clone https://github.com/jyhong0304/GeoDynFormer.git
cd GeoDynFormer
```

We recommend creating a dedicated Python environment before installing the dependencies.

For example:

```bash
conda create -n geodynformer python=3.7
conda activate geodynformer
pip install -r requirements.txt
```

---

## Train GeoDynFormer

GeoDynFormer models are trained separately for each subject.

### Prepare the training data

The training script expects each subject directory to contain a preprocessed
data file named:

```text
data_pre_split.pkl
```

For example:

```text
metadata_json/
└── s20/
    └── data_pre_split.pkl
```

### Weights & Biases

Training uses [Weights & Biases](https://wandb.ai/) for experiment logging.
Before starting a training run, log in once from the command line:

```bash
wandb login
```

### Train a subject-specific model

From the repository root, run:

```bash
python train_geodynformer.py \
    --raw_data_dir ./metadata_json/s20 \
    --save_dir ./saved_models_ats \
    --expt_name ats-s20 \
    --device cuda:0 \
    --wandb GeoDynFormer
```

In this example:

- `--raw_data_dir` specifies the subject directory containing
  `data_pre_split.pkl`;
- `--save_dir` specifies the parent directory for model outputs;
- `--expt_name` specifies the name of the training run;
- `--device` specifies the PyTorch device used for training; and
- `--wandb` specifies the Weights & Biases logging name used by the training
  interface.

The training outputs for this example are stored under:

```text
saved_models_ats/
└── ats-s20/
    ├── checkpoints/
    └── processed_data/
```

The default configuration is loaded from:

```text
affective_task/config/model_config.yaml
```

A different YAML configuration can be supplied with:

```bash
--config PATH_TO_CONFIG.yaml
```

### Optional training overrides

Selected training parameters can also be overridden directly from the command
line:

```text
--rand_seed
--batch_size
--upscale_mult
--max_epochs
```

For example:

```bash
python train_geodynformer.py \
    --raw_data_dir ./metadata_json/s20 \
    --save_dir ./saved_models_ats \
    --expt_name ats-s20 \
    --device cuda:0 \
    --wandb GeoDynFormer \
    --batch_size 32 \
    --max_epochs 2000
```

If these optional arguments are omitted, the corresponding values in
`model_config.yaml` are used.

## Reproduce the main manuscript figures

The main manuscript figures can be reproduced with:

```text
affective_task/plot/make_main.py
```

The script generates **Figures 2–8** in manuscript order using the pretrained
subject-specific GeoDynFormer models and the corresponding analysis outputs.

From the repository root, run:

```bash
python affective_task/plot/make_main.py \
    --model_dir ./trained_models \
    --figure_dir ./figures
```

Here:

- `--model_dir` specifies the directory containing the subject-specific model
  subdirectories;
- `--figure_dir` specifies where the generated figures are saved;
- `--patient_id` can optionally be used to provide a comma-separated subset of
  participant IDs; and
- `--do-preprocessing` can optionally be added to recompute the model-analysis
  outputs before generating the figures.

For example, to recompute the required model-analysis outputs and then generate
all main figures:

```bash
python affective_task/plot/make_main.py \
    --model_dir ./trained_models \
    --figure_dir ./figures \
    --do-preprocessing
```

If `--patient_id` is omitted, the script uses the full 22-participant manuscript
cohort.

The Figure 6f–g reliability analysis additionally expects the following
precomputed files under the repository-level `dataset/` directory:

```text
dataset/
├── trial_bootstrap_distribution.csv
├── trial_bootstrap_reliability_by_transition.csv
└── trial_bootstrap_reliability_summary.json
```

The plotting script saves the generated manuscript panels to the requested
figure directory and displays them interactively with Matplotlib.

## Reproduce the Extended Data figures

The Extended Data figures can be reproduced with:

```text
affective_task/plot/make_suppl.py
```

The script generates **Extended Data Figures 1–8** in manuscript order using the
pretrained subject-specific GeoDynFormer models and the corresponding analysis
outputs.

From the repository root, run:

```bash
python affective_task/plot/make_suppl.py \
    --model_dir ./trained_models \
    --figure_dir ./figures
```

Here:

- `--model_dir` specifies the directory containing the subject-specific model
  subdirectories;
- `--figure_dir` specifies where the generated figures are saved;
- `--patient_id` can optionally be used to provide a comma-separated subset of
  participant IDs; and
- `--do-preprocessing` can optionally be added to recompute the model-analysis
  outputs before generating the figures.

For example, to recompute the required model-analysis outputs and then generate
all Extended Data figures:

```bash
python affective_task/plot/make_suppl.py \
    --model_dir ./trained_models \
    --figure_dir ./figures \
    --do-preprocessing
```

If `--patient_id` is omitted, the script uses the full 22-participant manuscript
cohort.

The script generates the following figure modules in order:

```text
Extended Data Figure 1    figureS1.py
Extended Data Figure 2    figureS2.py
Extended Data Figure 3    figureS3.py
Extended Data Figure 4a-b figureS4AB.py
Extended Data Figure 5a   figureS5A.py
Extended Data Figure 5b   figureS5B.py
Extended Data Figure 6    figureS6.py
Extended Data Figure 7    figureS7.py
Extended Data Figure 8a   figureS8A.py
Extended Data Figure 8b   figureS8B.py
```

The plotting script saves the generated Extended Data panels to the requested
figure directory and displays them interactively with Matplotlib.

## Pretrained models

The trained models analyzed in the manuscript are publicly available through Zenodo:

https://doi.org/10.5281/zenodo.22776714

The underlying behavioral data are available from the corresponding author upon reasonable request, as described in the
manuscript.

## License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE.md) for details.

## Citation

If you use GeoDynFormer, please cite the associated bioRxiv preprint:

```bibtex
@article{hong2026geodynformer,
  author = {Hong, Jinyung and Nester, Elliot M. and Varisa, Lekha and
            Wrobel, Zuzanna and Rains, Kris Phataraphruk and Umesh, Tejas and
            Schneider, Tamera R. and Lisanby, Sarah H. and Roberts, Nicole A. and
            Turaga, Pavan and Brewer, Gene A. and Yang, Andrew I.},
  title = {Dynamics and geometry of emotion and cognition:
           an interpretable model of individual human behavior},
  journal = {bioRxiv},
  year = {2026},
  doi = {10.64898/2026.09.08.749516}
}
```

Citation information will be updated upon publication.

## Contact

For questions regarding the code or analyses:

**Jinyung Hong**  
Email: jinyung0304 [at] gmail [dot] com

For questions regarding the study:

**Andrew I. Yang**  
Department of Translational Neuroscience  
Barrow Neurological Institute  
Email: iyang.and [at] gmail [dot] com
