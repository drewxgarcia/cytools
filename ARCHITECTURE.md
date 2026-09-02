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

`cytools.__init__` is the public surface. Public names are listed in
`cytools.__all__` and resolved from an explicit lazy-export map. Importing the
package root therefore loads no numerical stack or domain module; accessing a
name imports only its owning module.

Upstream source compatibility is not a design constraint. A concept gets one
spelling; convenience aliases and renamed-parameter shims are deleted rather
than kept, because two names for one thing is a cost paid at every call site
and in every doc page. Removing or changing a public name requires a changelog
entry. Until the first tagged release it requires nothing further; after that,
a deprecation period applies.

The exception is a name dictated by a dependency. `Fan.cones` spells its index
flag `as_inds`, not the `as_indices` used everywhere else, because it overrides
`regfans.fan.Fan.cones` and renaming that parameter would break the override
contract. Such cases are documented at the definition, not worked around.

Core domain modules follow the same rule for expensive or cyclic peers.
`Polytope` imports `Triangulation` and `PolytopeFace` inside the operations that
construct them, while triangulation, toric-variety, and Calabi–Yau modules load
one another only at the operation boundary. Type-only dependencies belong
under `TYPE_CHECKING` with postponed annotations.

## Backend adapters

Each adapter owns dependency detection, dependency-specific calls, and output
normalization. It must not import `Polytope`, `Cone`, or another domain class.
For example, `Cone.hilbert_basis()` passes rays to
`cytools._backends.normaliz.hilbert_basis()` and caches the returned array; only
the adapter knows about PyNormaliz. Process-level compatibility work also lives
at this boundary: all PPL consumers import the engine through
`cytools._backends.ppl`, which restores the native floating-point rounding mode
once after the engine loads.

Mathematical tasks with interchangeable implementations are owned by an
engine registry. A domain call states the guarantees its result depends on
(for example, exact arithmetic, certified infeasibility, or regularity by
construction); the registry then selects the cheapest available engine that
provides all of them. Engine order is a measured performance policy, never a
correctness policy. Advanced users can scope a reproducibility or differential
test with `cytools.config.engines(...)`; overrides cannot silently weaken a
call site's guarantees.

`backend=` arguments remain at public entry points where they select a
genuinely different implementation, and they resolve into the same adapters.
They are not kept for compatibility: where such an argument had degenerated
into a no-op it was removed outright, as `Cone.is_pointed(backend, tol)` was.
New internal code must not thread backend strings through the domain graph or
import an optional implementation directly.

## Extension modules

Domain classes declare their entire method surface. Methods implemented by
`ntfe`, `vector_config`, or a focused helper module use an internal lazy
descriptor: the implementation module is loaded on first method access, while
the public call remains an ordinary bound method such as `poly.ntfe_frts()`.
Feature modules never mutate classes at import time, and the package root loads
their namespaces only when `cytools.ntfe` or `cytools.vector_config` is
requested explicitly. Each feature package defines a deliberate `__all__`;
package initializers must not use wildcard imports or leak implementation
dependencies into the supported namespace.

## Change checklist

- Put optional third-party APIs behind `_backends`.
- Resolve interchangeable implementations by required guarantees, not by a
  package name selected deep in a call chain.
- Keep imports and filesystem/network/process work out of module initialization.
- Declare methods on their owning classes; do not attach them from another
  module at runtime.
- Import internal names from their defining modules.
- Keep public exports explicit and covered by tests.
- Add correctness tests before moving behavior across module boundaries.
- Run `uv run pytest`; invoke benchmarks separately when performance changes.
