# Installing CYTools

Full documentation is available on the [CYTools website](https://cy.tools).

CYTools supports Python 3.10 or newer on Linux and Apple Silicon (M-series)
macOS. Intel-based Macs are not supported.

## Install from PyPI

Create an isolated environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install cytools
```

For the notebook-first interface, install the notebook extra and launch
JupyterLab:

```bash
python -m pip install "cytools[notebook]"
jupyter lab
```

Optional features are grouped by purpose:

| Extra | Adds |
| --- | --- |
| `notebook` | JupyterLab, widgets, and on-demand dataset access |
| `streaming` | On-demand dataset access without notebook dependencies |
| `gnn` | GNN-based triangulation sampling through `dualgnn` |
| `mosek` | The optional MOSEK solver |
| `normaliz` | In-process Hilbert-basis calculations through PyNormaliz |
| `performance` | CHOLMOD sparse solves through `scikit-sparse` |

Install one or more extras together, for example:

```bash
python -m pip install "cytools[gnn,notebook]"
```

The `performance` extra requires the SuiteSparse development libraries to be
available on the host system before `scikit-sparse` is installed.

## Develop from source

The repository uses [uv](https://docs.astral.sh/uv/) and commits its lockfile.
From a source checkout:

```bash
git clone https://github.com/LiamMcAllisterGroup/cytools.git
cd cytools
uv sync --extra notebook
uv run pytest
uv run jupyter lab
```

`uv sync` installs CYTools editably and includes the default development tools.
Add an optional feature with `uv sync --extra <name>`. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the test and benchmark workflow.

## Upgrade or uninstall

With the virtual environment activated:

```bash
python -m pip install --upgrade cytools
python -m pip uninstall cytools
```
