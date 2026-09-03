# Installing CYTools Workbench

CYTools Workbench is distributed as `cytools-workbench` but imported as
`cytools`. It replaces the official `cytools` distribution in an environment;
do not install both together.

The supported runtime is Python 3.12 or newer on Linux and macOS 15 or newer on
Apple Silicon (M-series). Intel-based Macs and Windows are not supported.

## Install from PyPI

Create an isolated environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install cytools-workbench
```

For the notebook-first interface, install the notebook extra and launch
JupyterLab:

```bash
python -m pip install "cytools-workbench[notebook]"
jupyter lab
```

Optional features are grouped by purpose:

| Extra | Adds |
| --- | --- |
| `notebook` | JupyterLab and notebook widgets |
| `streaming` | Explicit 4D database-shard download support |
| `cvxopt` | The optional CVXOPT quadratic-programming solver |
| `gnn` | GNN-based triangulation sampling through `dualgnn` |
| `mosek` | The optional MOSEK solver |
| `normaliz` | In-process Hilbert-basis calculations through PyNormaliz |
| `performance` | CHOLMOD sparse solves through `scikit-sparse` |

Install one or more extras together, for example:

```bash
python -m pip install "cytools-workbench[gnn,notebook]"
```

The `performance` extra requires the SuiteSparse development libraries to be
available on the host system before `scikit-sparse` is installed.

## Landscape data

Database reads never start a download. A vertex-count shard can be gigabytes,
and choosing a database snapshot is part of making a computation reproducible.
Point the workbench at an existing local snapshot before running a landscape
scan:

Set the snapshot location once for the current shell:

```bash
export CYTOOLS_DB_DIR=/path/to/polytopes-4d
```

The directory must contain files named
`polytopes-4d-05-vertices.parquet` through the vertex counts you intend to
query. Pass `db_dir=` to `scan` when notebook-local configuration is clearer.

If you explicitly want CYTools to fetch selected shards into the Hugging Face
cache, install the separate `streaming` extra and request the vertex counts:

```bash
python -m pip install "cytools-workbench[streaming]"
```

```python
from cytools import download_shards

download_shards([5, 6, 7])
```

## Develop from source

The repository uses [uv](https://docs.astral.sh/uv/) and commits its lockfile.
From a source checkout:

```bash
git clone https://github.com/drewxgarcia/cytools.git
cd cytools
uv sync --extra notebook
uv run pytest
uv run jupyter lab
```

`uv sync` installs CYTools Workbench editably and includes the default
development tools. Compiled backends remain opt-in so the default environment
is portable and does not combine incompatible native runtimes. Add a feature
with `uv sync --extra <name>`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
test and benchmark workflow.

The `performance` extra builds `scikit-sparse` from source and needs the
SuiteSparse headers first (`brew install suite-sparse` on macOS,
`libsuitesparse-dev` on Debian/Ubuntu).

### Duplicate OpenMP runtimes on macOS

Installing both `gnn` and `performance` can put two OpenMP runtimes in one
process: homebrew's SuiteSparse links `/opt/homebrew/opt/libomp/lib/libomp.dylib`,
while PyTorch bundles its own under `torch/lib/`. LLVM's runtime calls
`abort()` rather than tolerating a duplicate, so the interpreter can die with
SIGABRT and no Python traceback.

CYTools refuses to import PyTorch when it detects this collision, naming both
paths instead of allowing LLVM to abort the interpreter. Keep the extras in
separate environments, or make PyTorch share the runtime already present:

```bash
ln -sf /opt/homebrew/opt/libomp/lib/libomp.dylib \
       "$(uv run python -c 'import torch,pathlib;print(pathlib.Path(torch.__file__).parent/"lib"/"libomp.dylib")')"
```

Re-apply it after any reinstall of PyTorch, which restores the bundled copy.
`KMP_DUPLICATE_LIB_OK=TRUE` silences the abort instead, but it suppresses the
guard rather than removing the duplicate and is not safe for numerical work.

## Upgrade or uninstall

With the virtual environment activated:

```bash
python -m pip install --upgrade cytools-workbench
python -m pip uninstall cytools-workbench
```
