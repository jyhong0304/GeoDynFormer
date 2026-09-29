# Pretrained Models

This directory is used to store pretrained GeoDynFormer models required to reproduce the analyses and figures in this repository.

The pretrained model files are **not included directly in the GitHub repository** because of their size. They can be downloaded separately from Zenodo:

**Zenodo:** [https://doi.org/10.5281/zenodo.22776714](https://doi.org/10.5281/zenodo.22776714)

## Setup

1. Download the pretrained model archive from the Zenodo link above.
2. Extract the downloaded files.
3. Place the extracted model files inside this `trained_models/` directory.

After extraction, the directory should follow the structure provided in the Zenodo archive.

For example:

```text
GeoDynFormer/
├── affective_task_switching/
├── dataset/
├── trained_models/
│   ├── s20
│   └── ...
└── figures/
```

The analysis and visualization scripts in this repository assume that the pretrained models are available under this directory.

## Notes

- Do not rename the downloaded model files or directories unless otherwise specified.
- Large model checkpoint files are excluded from Git version control.
- Please refer to the main repository `README.md` for instructions on reproducing the analyses and figures.