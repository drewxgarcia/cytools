
<p align="center">
    <img src="https://cy.tools/img/titleimage-circle.png?sanitize=true" width="250"></img><br></br>
    <b>A software package for analyzing Calabi-Yau manifolds</b><br></br>
    <img alt="Latest release" src="https://img.shields.io/github/v/release/liammcallistergroup/cytools"></img>
    <img alt="Number of downloads" src="https://img.shields.io/github/downloads/liammcallistergroup/cytools/total"></img>
    <img alt="License" src="https://img.shields.io/github/license/liammcallistergroup/cytools"></img>
</p>

-------------------------------------------------------------------------------

CYTools is an open-source software package developed by [Liam McAllister's group](https://liammcallistergroup.com/) with the purpose of studying Calabi-Yau manifolds arising from the Kreuzer-Skarke database. The founding authors are [Mehmet Demirtas](https://inspirehep.net/authors/1765325) and [Andres Rios-Tascon](https://ariostas.com); the current [BDFL](https://en.wikipedia.org/wiki/Benevolent_dictator_for_life) is [Nate MacFadden](https://inspirehep.net/authors/1590972). It emerged from several years of effort towards exploring previously uncharted parts of the string landscape. It offers vastly superior computational performance compared to other software that are typically used in the field, as discussed in the CYTools paper [arXiv:2211.03823](https://arxiv.org/abs/2211.03823). Installation instructions and detailed documentation can be found in the [CYTools website](https://cy.tools).

Most of the code is written in Python, with wrappers to interface with various other open-source software. It is distributed as a Python package through PyPI, with a locked `uv` environment for reproducible source development.

## Features

* **Polytopes and triangulations.** Lattice point enumeration, face lattices, cone computations, and fine regular star triangulations, along with utilities to fetch reflexive polytopes from the Kreuzer-Skarke database (`cytools.fetch_polytopes`).
* **Calabi-Yau hypersurfaces.** Hodge numbers, intersection numbers, Mori and Kähler cones, divisor and Calabi-Yau volumes, second Chern class, and Gopakumar-Vafa invariants.
* **NTFE enumeration.** Enumeration and sampling of the expanded secondary cones and the corresponding FR(S)Ts of a polytope, following [arXiv:2309.10855](https://arxiv.org/abs/2309.10855) (`cytools.ntfe`).
* **GNN triangulation sampling.** Near-uniform sampling of NTFE FR(S)Ts using the dualGNN graph neural network ([arXiv:2605.27770](https://arxiv.org/abs/2605.27770)) to sample the 2-face triangulations (`Polytope.random_triangulations_gnn`). This requires the optional `dualgnn` package: `pip install "cytools[gnn]"`.
* **F-theory tooling.** Orientifolds and F-theory uplifts of Calabi-Yau hypersurfaces (`cytools.f_theory`).
* **Notebook-first landscape scans.** Query database columns into pandas, lazily compute derived geometry, and resume cached scans by stable Kreuzer-Skarke IDs (`cytools.scan`, `cytools.sweep`).

## Quick example

After [installing CYTools](INSTALL.md), compute the Hodge numbers of the quintic Calabi-Yau threefold:

```python
from cytools import Polytope

p = Polytope([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]])
cy = p.triangulate().get_cy()
print(cy.h11(), cy.h21())   # 1 101
```

## Notebook-first landscape scans

The landscape API is currently part of the unreleased development tree. From a
source checkout, install the notebook and on-demand dataset support with:

```bash
uv sync --extra notebook
```

After the next CYTools release, the equivalent command will be
`pip install "cytools[notebook]"`. Start with a small vertex-count range so the
first download stays small:

```python
from cytools import quantities, scan

quantities()  # every built-in column and whether it can run in parallel

df = scan(
    ["h11", "h21", "chi", "n_points"],
    n=100,
    n_vertices=[5, 6, 7],
    stream=True,
)
df.head()
```

Database-backed columns are read directly from Parquet without constructing a
`Polytope`. Derived columns build only the geometry they need and are cached by
stable `ks_id`, so repeating the same call resumes immediately:

```python
df = scan(
    ["h11", "is_favorable", "n_intnums"],
    n=1_000,
    n_vertices=[5, 6, 7],
    stream=True,
)
```

Volume scans default to the tip of the stretched Kähler cone. For an ensemble
with one reproducible interior direction per geometry, select `moduli="sampled"`
and retain the point alongside the resulting volumes. Sampled rays are rescaled
to the same minimum curve-volume convention as the tip:

```python
df = scan(
    ["h11", "kahler_point", "divisor_volumes", "cy_volume"],
    n=1_000,
    moduli="sampled",
)
```

The mode is recorded in `df.attrs["cytools"]`; tip and sampled results use
separate cache keys.

The high-level API consistently uses CYTools' N-lattice convention, including
for `h11`, `h21`, and `chi` filters. A capped query is a reproducible,
bounded-memory stratified sample across files and shuffled Parquet row groups;
it is not a uniform sample of every matching database row.

Notebook-defined columns use the same interface. They run safely in the
notebook process; bump `version` when their meaning changes:

```python
from cytools import quantity

@quantity
def max_vertex_coordinate(g):
    """Largest absolute coordinate among the vertices."""
    return abs(g.polytope.vertices()).max()

df = scan(
    ["h11", "max_vertex_coordinate"],
    n=250,
    n_vertices=[5, 6, 7],
    stream=True,
    version=1,
)
```

For runs too large to collect into one DataFrame, `sweep(...)` computes and
stores results with bounded memory and returns progress counts. Use `status()`
to inspect the cache. Local database users can omit `stream=True` and set
`CYTOOLS_DB_DIR`, or pass `db_dir=` directly. See the executable
[landscape notebook](demos/landscape_scans.ipynb) for the complete workflow.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and
[ARCHITECTURE.md](ARCHITECTURE.md) for package boundaries and design rules.

## Citation

If you use CYTools in your work, please cite the CYTools paper, *CYTools: A Software Package for Analyzing Calabi-Yau Manifolds* ([arXiv:2211.03823](https://arxiv.org/abs/2211.03823)). Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

## Acknowledgements

CYTools makes use of a variety of open-source projects. It includes a few code snippets from [SageMath](https://www.sagemath.org/) [[GPLv2](http://www.gnu.org/licenses/gpl-2.0.html)], a modified version of [TOPCOM](https://www.wm.uni-bayreuth.de/de/team/rambau_joerg/TOPCOM/index.html) [[GPLv2](http://www.gnu.org/licenses/gpl-2.0.html)] that can be found [here](https://github.com/LiamMcAllisterGroup/topcom), the [Computational Geometry Algorithms Library](https://www.cgal.org) [[LGPLv3](http://www.gnu.org/licenses/lgpl-3.0.html)], and multiple Python packages including [SciPy](https://www.scipy.org/), [NumPy](https://numpy.org/), [pplpy](https://gitlab.com/videlec/pplpy), [OR-Tools](https://developers.google.com/optimization), [Normaliz](https://github.com/Normaliz/Normaliz), [scikit-sparse](https://github.com/scikit-sparse/scikit-sparse), and [flint-py](https://gitlab.com/alisianoi/flint-py).

All original CYTools code is distributed under the terms of the [GNU General Public License version 3](https://www.gnu.org/licenses/gpl-3.0.txt). All other packages and code snippets are redistributed under their respective licenses.

Questions, comments and/or suggestions can be directed to [support@cy.tools](mailto:support@cy.tools).
