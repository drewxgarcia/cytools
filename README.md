
<p align="center">
    <img src="https://cy.tools/img/titleimage-circle.png?sanitize=true" width="250"></img><br></br>
    <b>CYTools Workbench — notebook-first Calabi-Yau landscape studies</b><br></br>
    <img alt="Latest release" src="https://img.shields.io/github/v/release/drewxgarcia/cytools"></img>
    <img alt="License" src="https://img.shields.io/github/license/drewxgarcia/cytools"></img>
</p>

-------------------------------------------------------------------------------

CYTools Workbench is an independently maintained, notebook-first fork of
[CYTools](https://github.com/LiamMcAllisterGroup/cytools), the open-source
Calabi-Yau geometry package developed by Liam McAllister's group. It keeps the
compatible `cytools` import namespace while adding reproducible, resumable
landscape scans and a modern data-analysis workflow.

Install the fork as `cytools-workbench`; it is a drop-in replacement and must
not be installed in the same environment as the official `cytools`
distribution. The fork has its own versions and release lifecycle. General
fixes are kept separable so they can be proposed upstream, while the notebook
and landscape product surface can evolve independently. The current upstream
base is available at runtime as `cytools.upstream_version`.

## Features

* **Polytopes and triangulations.** Lattice point enumeration, face lattices, cone computations, and fine regular star triangulations, along with utilities to fetch reflexive polytopes from the Kreuzer-Skarke database (`cytools.fetch_polytopes`).
* **Calabi-Yau hypersurfaces.** Hodge numbers, intersection numbers, Mori and Kähler cones, divisor and Calabi-Yau volumes, second Chern class, and Gopakumar-Vafa invariants.
* **NTFE enumeration.** Enumeration and sampling of the expanded secondary cones and the corresponding FR(S)Ts of a polytope, following [arXiv:2309.10855](https://arxiv.org/abs/2309.10855) (`cytools.ntfe`).
* **GNN triangulation sampling.** Near-uniform sampling of NTFE FR(S)Ts using the dualGNN graph neural network ([arXiv:2605.27770](https://arxiv.org/abs/2605.27770)) to sample the 2-face triangulations (`Polytope.random_triangulations_gnn`). This requires the optional `dualgnn` package: `pip install "cytools-workbench[gnn]"`.
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

From a source checkout, install the notebook environment with:

```bash
uv sync --extra notebook
```

For a release install, use `pip install "cytools-workbench[notebook]"`. Start
with a small vertex-count range: the first scan downloads only those 4D
Parquet shards and later scans reuse the local Hugging Face cache. For large or
fully pinned studies, set `CYTOOLS_DB_DIR` to an explicit database snapshot.

```python
from cytools import quantities, scan

quantities()  # every built-in column and whether it can run in parallel

df = scan(
    ["h11", "h21", "chi", "n_points"],
    n=100,
    n_vertices=[5, 6, 7],
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
    version=1,
)
```

For runs too large to collect into one DataFrame, `sweep(...)` computes and
stores results with bounded memory and returns progress counts. Use `status()`
to inspect the cache. Set `CYTOOLS_DB_DIR`, or pass `db_dir=` directly, to
prefer an explicit local snapshot. See the executable
[landscape notebook](demos/landscape_scans.ipynb) for the complete workflow.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and
[ARCHITECTURE.md](ARCHITECTURE.md) for package boundaries and design rules.

## Citation

If you use CYTools in your work, please cite the CYTools paper, *CYTools: A Software Package for Analyzing Calabi-Yau Manifolds* ([arXiv:2211.03823](https://arxiv.org/abs/2211.03823)). Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

## Acknowledgements

CYTools makes use of a variety of open-source projects. It includes a few code snippets from [SageMath](https://www.sagemath.org/) [[GPLv2](http://www.gnu.org/licenses/gpl-2.0.html)], a modified version of [TOPCOM](https://www.wm.uni-bayreuth.de/de/team/rambau_joerg/TOPCOM/index.html) [[GPLv2](http://www.gnu.org/licenses/gpl-2.0.html)] that can be found [here](https://github.com/LiamMcAllisterGroup/topcom), the [Computational Geometry Algorithms Library](https://www.cgal.org) [[LGPLv3](http://www.gnu.org/licenses/lgpl-3.0.html)], and multiple Python packages including [SciPy](https://www.scipy.org/), [NumPy](https://numpy.org/), [pplpy](https://gitlab.com/videlec/pplpy), [OR-Tools](https://developers.google.com/optimization), [Normaliz](https://github.com/Normaliz/Normaliz), [scikit-sparse](https://github.com/scikit-sparse/scikit-sparse), and [flint-py](https://gitlab.com/alisianoi/flint-py).

All original CYTools code is distributed under the terms of the [GNU General Public License version 3](https://www.gnu.org/licenses/gpl-3.0.txt). All other packages and code snippets are redistributed under their respective licenses.

Questions, bug reports, and product suggestions belong in this fork's
[issue tracker](https://github.com/drewxgarcia/cytools/issues). Issues that also
affect official CYTools can be distilled into an upstream report or pull
request.
