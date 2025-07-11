# Crash Casino Game Analysis

This repository contains experiments for clustering variable-length time series from the Crash game.

## Data cleaning

Some scraped rounds contain artifacts in the final tick due to optical character recognition errors.
The script `src/data_cleaning.py` provides utilities to clean these CSV files. It corrects misplaced
decimal points and can drop clearly corrupted final values.

Run it as:

```bash
python -m src.data_cleaning <input_dir> <output_dir>
```

Cleaned files will be written to `<output_dir>` preserving the original directory structure.

## Running the sample analysis

The exploratory analysis script uses multiprocessing and expects the `src` package
to be importable. Run it as a module from the project root:

```bash
python -m src.main
```

This ensures that subprocesses can properly import the package and prevents
`ModuleNotFoundError` errors on platforms that use the spawn start method.
