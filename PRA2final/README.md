# Task 1 Code Group 2

## Run

Create a venv if you want, install deps, then run from this folder:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_task1.py --data-root Data-PR-As2 --dataset all
```

Use `--dataset image` or `--dataset genes` for a single pipeline. `--data-root` is joined to the parent of this folder so pick the path that actually reaches your dataset (for example `PRA2final/Data-PR-As2` if the data lives inside the repo on disk). Random seed is fixed to `42` in `run_task1.py`.

## What each file does

`run_task1.py` parses `--data-root` and `--dataset`, sets the seed and calls the image and/or gene pipeline in order.

`src/data_loading.py` finds the image tree, loads images and gene CSVs and writes basic summaries and class distribution plots.

`src/image_pipeline.py` holds the image workflow feature sweep, resizing or reduced features, classification, clustering, validation comparison, augmentation vs baseline and ensemble.

`src/gene_pipeline.py` holds the gene workflow feature sweep, classification, clustering, validation comparison and ensemble.

`src/evaluation.py` computes classification metrics, confusion matrices and clustering scores used by both workflows.

`src/visualization.py` saves figures such as bar charts, line plots, scatter plots and confusion matrices.

`src/utils.py` sets seeds, creates output directories and writes JSON for saved hyperparameters.

Results from a run go under `outputs/`.

## Authors

- Eli Baho (s4807103)
- Marco Capoccia (s4807189)
- Floris Pol (s4909925)
- Andy Slettenhaar (s4781627)
