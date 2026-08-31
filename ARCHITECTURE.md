# CYTools architecture

CYTools is organized around a small public facade, a domain model for toric
geometry, workflow modules for large computations, and adapters for external
engines. The boundaries below are intentionally stricter than the historical
file layout: they describe where new code belongs and how existing code should
be migrated.

```text
cytools.__init__                 public names only
    ├── domain objects          Polytope, Cone, Triangulation, ...
    │     ├── helpers           array and geometry primitives
    │     └── _backends         third-party engine adapters
    ├── landscape              notebook-facing orchestration
    │     ├── dataset           source scanning and decoding
    │     └── store             derived-result persistence
    └── feature modules        ntfe, vector_config, f_theory
```

## Dependency rules

1. Internal modules import the concrete module that owns a name; they do not
   import from `cytools`, which is the public facade.
2. Domain objects do not expose objects from computational engines. Modules in
   `cytools._backends` translate plain arrays into engine calls and return plain
   arrays or Python values.
3. Optional engines are imported inside the adapter operation that needs them.
   Importing `cytools` must not load an optional extension, start a process,
   access the network, or create files.
4. Dataset and persistence code do not own geometry algorithms. Landscape
   orchestration requests geometry through the domain API and stores only
   stable, serializable results.
5. Benchmarks and their datasets live under `benchmarks/`; they are not part of
   the installed package.

These rules are checked in `tests/test_architecture.py` where they can be
verified mechanically.

## Public API

`cytools.__init__` is the compatibility surface. A name is public only when it
is imported there and listed in `cytools.__all__`. Internal modules may change
without a deprecation cycle; removing or changing a public name requires a
changelog entry and, after a release, a deprecation period.

## Backend adapters

Each adapter owns dependency detection, dependency-specific calls, and output
normalization. It must not import `Polytope`, `Cone`, or another domain class.
For example, `Cone.hilbert_basis()` passes rays to
`cytools._backends.normaliz.hilbert_basis()` and caches the returned array; only
the adapter knows about PyNormaliz.

## Extension modules

Domain classes declare their entire method surface. Methods implemented by
`ntfe`, `vector_config`, or a focused helper module use an internal lazy
descriptor: the implementation module is loaded on first method access, while
the public call remains an ordinary bound method such as `poly.ntfe_frts()`.
Feature modules never mutate classes at import time, and the package root loads
their namespaces only when `cytools.ntfe` or `cytools.vector_config` is
requested explicitly.

## Change checklist

- Put optional third-party APIs behind `_backends`.
- Keep imports and filesystem/network/process work out of module initialization.
- Declare methods on their owning classes; do not attach them from another
  module at runtime.
- Import internal names from their defining modules.
- Keep public exports explicit and covered by tests.
- Add correctness tests before moving behavior across module boundaries.
- Run `uv run pytest`; invoke benchmarks separately when performance changes.
