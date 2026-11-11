# Handwritten Sudoku Reader and Solver (Project P7)

An integrated Intelligent Systems project combining a **calibrated PyTorch neural network** (`DigitMLP`), **correction search** (`UCS`, `Greedy Best-First`, `A*`), and a **Sudoku CSP solver** (`Backtracking`, `Forward Checking`, `AC-3`) to read handwritten `252x252` MNIST Sudoku mosaics, detect and correct misread digits, and recover the true solution.

## Folder Layout
- `data/`: MNIST download/splitting (`mnist_loader.py`), shared label mapping and brightness blank rule (`labels.py`), seeded Sudoku puzzle/mosaic generator (`sudoku_generator.py`), and [DATA_DICTIONARY.md](data/DATA_DICTIONARY.md).
- `nn/`: PyTorch MLP classifier and temperature scaling (`model.py`), training and Logistic Regression baseline comparison (`train.py`).
- `csp/`: 81-variable Sudoku CSP solver with switchable `use_fc` and `use_ac3` (`sudoku_csp.py`).
- `search/`: Correction search (`UCS`, `Greedy`, admissible `A*`) with timeout/node limits and diagnostics (`correction_search.py`, `diagnostic.py`).
- `integration/`: `252x252` mosaic reading (`reader.py`) and end-to-end solving pipeline (`pipeline.py`).
- `eval/`: Multi-seed experiments 1–3 (`run_experiments.py`).
- `tests/`: Automated `pytest` suite covering normal, edge, and impossible cases.
- `results/`: Saved metrics, CSV/Markdown tables, plots, and [NOTES.md](results/NOTES.md).
- `make_data.py`: Reproducible MNIST preprocessing script.
- `run_demo.py`: Interactive end-to-end demonstration and cooperation plots.

## Environment Setup (Windows PowerShell)

Create and activate a Python virtual environment, then install pinned dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

> Note: `requirements.txt` pins CPU PyTorch (`torch==2.15.0+cpu`, `torchvision==0.29.2+cpu`) and includes `--extra-index-url https://download.pytorch.org/whl/cpu` so `pip install -r requirements.txt` works directly.

## Exact Run Commands

### 1. Prepare Dataset (MNIST digits 1–9, 50,000 train / 4,077 val / 9,020 test)
```powershell
.venv\Scripts\python.exe make_data.py --seed 42
```

### 2. Train Neural Network + Temperature Scaling + Logistic Regression Baseline
```powershell
.venv\Scripts\python.exe -m nn.train --seed 42
```

### 3. Run Integrated Demo
```powershell
# Default 2-misread solved demo with search trace and plots in results/demo/
.venv\Scripts\python.exe run_demo.py --seed 43 --misreads 2 --algo astar --trace

# Clean puzzle (0 misreads) and 1-misread puzzle
.venv\Scripts\python.exe run_demo.py --seed 42 --misreads 0
.venv\Scripts\python.exe run_demo.py --seed 42 --misreads 1

# Skip NN-guided correction search (--no-nn)
.venv\Scripts\python.exe run_demo.py --seed 43 --misreads 2 --no-nn
```

### 4. Run Unit & Integration Tests
```powershell
.venv\Scripts\python.exe -m pytest -v
```

### 5. Run Multi-Seed Experiments (Results saved to `results/`)
```powershell
# Experiment 1: UCS vs Greedy vs A* (k=0..4, require_unique=False & True)
.venv\Scripts\python.exe -m eval.run_experiments --exp 1 --seeds 42 43 44 45 46 --puzzles 20 --timeout 2.5

# Experiment 2: BT vs BT+FC vs BT+FC+AC-3 (25, 30, 35 givens)
.venv\Scripts\python.exe -m eval.run_experiments --exp 2 --seeds 42 43 44 45 46 --csp-puzzles 20 --timeout 2.5

# Experiment 3: NN-guided vs No-correction vs Random-correction (k=0..4)
.venv\Scripts\python.exe -m eval.run_experiments --exp 3 --seeds 42 43 44 45 46 --puzzles 20 --timeout 2.5

# Run all three experiments together (use --puzzles 100 for 100 puzzles/level)
.venv\Scripts\python.exe -m eval.run_experiments --exp all --seeds 42 43 44 45 46 --puzzles 20 --csp-puzzles 20 --timeout 2.5
```

