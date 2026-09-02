# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# CYTools is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# CYTools. If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Engine resolution. Guarantees decide eligibility;
#               measurements order eligible implementations.
#
#               A *task* is a mathematical problem (build a convex hull, find
#               an interior point). An *engine* is one implementation of a
#               task, which declares the guarantees it provides and whether
#               its dependency is importable. A call site states the
#               guarantees its mathematics requires; the registry returns the
#               cheapest available engine that provides them.
#
#               This replaces a `backend=` string threaded through the public
#               API. That parameter asked the caller to make a choice that is
#               not theirs to make: `Cone.is_solid` reads a None return as
#               "not full-dimensional", so an optimizer that cannot certify
#               infeasibility does not merely run slower there, it returns a
#               different answer. Guarantees are properties of the call site,
#               so the call site is where they belong.
# -----------------------------------------------------------------------------

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "COMPLETE",
    "CERTIFIES_INFEASIBLE",
    "DETERMINISTIC",
    "EXACT",
    "RECOVERABLE",
    "REGULAR",
    "Engine",
    "EngineUnavailable",
    "GuaranteeViolation",
    "Registry",
    "UnknownEngine",
    "get_overrides",
    "override",
    "set_overrides",
]


# guarantees
# ==========
# Declared by engines, required by call sites. These are deliberately about
# *what is computed*, never about speed: an engine that is merely slower is
# ordered later, while an engine that provides a weaker guarantee is excluded
# outright.

#: The result is computed in exact integer or rational arithmetic. No
#: floating-point rounding step stands between the input and the output. For a
#: reflexive lattice polytope a rounded facet is not a slower answer, it is a
#: different polytope.
EXACT = "exact"

#: A ``None`` return is an exact proof that no solution exists, not merely a
#: floating-point solver status or an exhausted search bound. Required wherever
#: absence of a solution is read as a mathematical conclusion.
CERTIFIES_INFEASIBLE = "certifies_infeasible"

#: The triangulation produced is regular by construction, so regularity needs
#: no separate verification and cannot silently fail to hold.
REGULAR = "regular"

#: The engine enumerates the entire solution set rather than returning one
#: representative element.
COMPLETE = "complete"

#: Repeated runs on identical input produce identical output. Engines that
#: perturb their input with random noise do not provide this.
DETERMINISTIC = "deterministic"

#: Failure arrives as a Python exception, not as process death.
#:
#: This is not hypothetical. PALP is a C program with compile-time array
#: bounds (`CEQ_Nmax`, `EQUA_Nmax`) and calls `abort()` when a configuration
#: exceeds them or contains duplicate points. Reached through a Python
#: extension that is SIGABRT, killing the interpreter with no traceback and
#: taking any unsaved notebook state with it. Measured on this tree: a
#: 9-dimensional configuration of 40 points is enough. Required by every call
#: site reachable from user input with unbounded size.
RECOVERABLE = "recoverable"

_ALL_GUARANTEES = frozenset(
    {EXACT, CERTIFIES_INFEASIBLE, REGULAR, COMPLETE, DETERMINISTIC, RECOVERABLE}
)


# errors
# ======
class EngineUnavailable(RuntimeError):
    """No engine registered for a task can satisfy the requested guarantees.

    Carries the per-engine reason for each rejection, because "no engine
    available" without the reasons is not actionable: a missing optional
    dependency and an unsatisfiable guarantee need different fixes.
    """


class UnknownEngine(ValueError):
    """An override named an engine that is not registered for the task."""


class GuaranteeViolation(RuntimeError):
    """An override named an engine providing weaker guarantees than required.

    Permitted, with a warning, under ``allow_weaker=True``. Differential tests
    that compare a strong engine against a weak one are the legitimate use.
    """


# overrides
# =========
# A ContextVar rather than a module global: it is correct under threads and
# under asyncio, and it restores cleanly on exit even if the body raises.
#
# It does *not* cross a process boundary. Worker processes start with an empty
# mapping, which is the safe default (they resolve on their own capabilities).
# A pool that must inherit the parent's overrides should pass get_overrides()
# to its workers and apply it with set_overrides() in the initializer.
_overrides: ContextVar[Mapping[str, str]] = ContextVar(
    "cytools_engine_overrides", default={}
)


def get_overrides() -> dict[str, str]:
    """The engine overrides active in this context, as a plain dict."""
    return dict(_overrides.get())


def set_overrides(mapping: Mapping[str, str]) -> None:
    """Replace the active overrides. Intended for worker-process initializers."""
    _overrides.set(dict(mapping))


@dataclass
class _OverrideContext:
    """Context manager restoring the previous overrides on exit."""

    mapping: dict[str, str]
    allow_weaker: bool
    _token: Any = field(default=None, init=False, repr=False)

    def __enter__(self) -> _OverrideContext:
        merged = dict(_overrides.get())
        merged.update(self.mapping)
        if self.allow_weaker:
            merged["*allow_weaker*"] = "1"
        else:
            merged.pop("*allow_weaker*", None)
        self._token = _overrides.set(merged)
        return self

    def __exit__(self, *exc: object) -> None:
        _overrides.reset(self._token)


def override(*, allow_weaker: bool = False, **engines: str) -> _OverrideContext:
    """
    **Description:**
    Force specific engines for the duration of a ``with`` block. Public entry
    point is :func:`cytools.config.engines`.

    **Arguments:**
    - `allow_weaker`: Permit an engine whose guarantees are weaker than the
        call site requires, downgrading the error to a warning.
    - `**engines`: Task name to engine name.

    **Returns:**
    A context manager.
    """
    return _OverrideContext(dict(engines), allow_weaker)


# engines and registries
# ======================
@dataclass(frozen=True)
class Engine:
    """One implementation of a task.

    **Attributes:**
    - `name`: Stable identifier, used by overrides and in diagnostics.
    - `run`: The callable. Every engine in a registry shares one signature.
    - `provides`: Guarantees this engine makes. Anything absent is assumed not
        to hold, so a new engine defaults to the weakest possible claim.
    - `available`: Whether the engine's dependency can be used in this
        process. Called at resolution time, never at import time, so an
        optional dependency is not imported merely by loading this module.
    - `applies`: Whether the engine supports a particular problem at all.
        This is a hard capability boundary, never a performance heuristic.
    - `preference`: A problem-dependent cost rank. Lower is preferred. This is
        where measured size crossovers live; it may reorder engines but never
        make an otherwise valid explicit selection illegal.
    - `why_unavailable`: Human-readable reason, shown when resolution fails.
    """

    name: str
    run: Callable[..., Any]
    provides: frozenset[str] = frozenset()
    available: Callable[[], bool] = lambda: True
    applies: Callable[[Mapping[str, Any]], bool] = lambda problem: True
    preference: Callable[[Mapping[str, Any]], float] = lambda problem: 0
    why_unavailable: str = "the engine is unavailable in this environment"

    def __post_init__(self) -> None:
        unknown = frozenset(self.provides) - _ALL_GUARANTEES
        if unknown:
            raise ValueError(
                f"Engine {self.name!r} declares unknown guarantees: "
                f"{sorted(unknown)}. Known guarantees: {sorted(_ALL_GUARANTEES)}."
            )


@dataclass(frozen=True)
class Registry:
    """The engines implementing one task, in stable tie-break order.

    Problem-dependent preference scores order qualifying engines; declaration
    order breaks ties. Neither is a correctness mechanism: an engine that
    cannot provide a required guarantee is excluded regardless of rank.
    """

    task: str
    engines: tuple[Engine, ...]

    def __post_init__(self) -> None:
        names = [e.name for e in self.engines]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate engine names in task {self.task!r}: {names}")

    # introspection
    # -------------
    def __getitem__(self, name: str) -> Engine:
        for engine in self.engines:
            if engine.name == name:
                return engine
        raise UnknownEngine(
            f"No engine {name!r} for task {self.task!r}. "
            f"Registered: {[e.name for e in self.engines]}."
        )

    def names(self) -> tuple[str, ...]:
        """Every engine name, in stable registration order."""
        return tuple(e.name for e in self.engines)

    def available(self) -> tuple[str, ...]:
        """The usable engine names, in stable registration order."""
        return tuple(e.name for e in self.engines if e.available())

    def select(self, name: str, problem: Mapping[str, Any] | None = None) -> Engine:
        """Select an explicitly named engine without inventing requirements.

        Historical ``backend=`` parameters already represent an explicit
        implementation choice. They use this method to retain that contract
        while sharing registry diagnostics and adapters. New automatic call
        sites should use :meth:`resolve` and state the guarantees their
        mathematics requires.
        """
        problem = {} if problem is None else problem
        engine = self[name]
        if not engine.available():
            raise EngineUnavailable(
                f"Engine {name!r} was selected for task {self.task!r}, but "
                f"{engine.why_unavailable}."
            )
        if not engine.applies(problem):
            raise EngineUnavailable(
                f"Engine {name!r} was selected for task {self.task!r}, but it "
                f"does not support problem {dict(problem)}."
            )
        return engine

    # resolution
    # ----------
    def candidates(
        self,
        need: Iterable[str] = (),
        problem: Mapping[str, Any] | None = None,
    ) -> tuple[Engine, ...]:
        """
        **Description:**
        Every qualifying engine, cheapest first, rather than only the best.

        For call sites that validate their own result and can retry: a
        numerically failed factorization is not a wrong answer, so falling
        through to the next engine is correct there. Call sites that cannot
        check their result must use `resolve` instead.

        **Arguments:**
        - `need`: Guarantees the call site requires.
        - `problem`: Problem descriptor for the `applies` predicates.

        **Returns:**
        The qualifying engines, in preference order. Honours an active
        override by returning just that engine.
        """
        need = frozenset(need)
        problem = {} if problem is None else problem
        active = _overrides.get()
        if active.get(self.task) is not None:
            return (self.resolve(need, problem),)
        eligible = tuple(
            e
            for e in self.engines
            if not (need - e.provides) and e.available() and e.applies(problem)
        )
        order = {engine.name: i for i, engine in enumerate(self.engines)}
        return tuple(
            sorted(eligible, key=lambda e: (e.preference(problem), order[e.name]))
        )

    def resolve(
        self,
        need: Iterable[str] = (),
        problem: Mapping[str, Any] | None = None,
    ) -> Engine:
        """
        **Description:**
        The cheapest available engine providing every guarantee in `need`.

        **Arguments:**
        - `need`: Guarantees the call site's mathematics depends on. An engine
            that does not provide all of them is excluded, not merely
            deprioritised.
        - `problem`: Problem descriptor consulted by each engine's `applies`
            predicate; typically carries `dim` and/or `size`.

        **Returns:**
        The selected `Engine`.

        **Raises:**
        `EngineUnavailable` when nothing qualifies, listing why each candidate
        was rejected. `UnknownEngine` or `GuaranteeViolation` for a bad
        override.
        """
        need = frozenset(need)
        unknown = need - _ALL_GUARANTEES
        if unknown:
            raise ValueError(
                f"Unknown guarantee(s) required for task {self.task!r}: "
                f"{sorted(unknown)}."
            )
        problem = {} if problem is None else problem

        active = _overrides.get()
        forced = active.get(self.task)
        if forced is not None:
            return self._resolve_forced(forced, need, problem, active)

        rejected: list[str] = []
        eligible: list[tuple[float, int, Engine]] = []
        for index, engine in enumerate(self.engines):
            missing = need - engine.provides
            if missing:
                rejected.append(f"{engine.name}: does not provide {sorted(missing)}")
                continue
            if not engine.available():
                rejected.append(f"{engine.name}: {engine.why_unavailable}")
                continue
            if not engine.applies(problem):
                rejected.append(f"{engine.name}: does not support this problem")
                continue
            eligible.append((engine.preference(problem), index, engine))

        if eligible:
            return min(eligible, key=lambda item: (item[0], item[1]))[2]

        raise EngineUnavailable(
            f"No engine for task {self.task!r} provides {sorted(need)} "
            f"for problem {dict(problem)}.\n  " + "\n  ".join(rejected)
        )

    def _resolve_forced(
        self,
        name: str,
        need: frozenset[str],
        problem: Mapping[str, Any],
        active: Mapping[str, str],
    ) -> Engine:
        """Apply an override, checking availability and guarantees."""
        engine = self.select(name, problem)
        missing = need - engine.provides
        if missing:
            message = (
                f"Engine {name!r} was forced for task {self.task!r}, but this "
                f"call requires {sorted(missing)}, which it does not provide. "
                "The result may be mathematically wrong, not merely slower."
            )
            if active.get("*allow_weaker*"):
                warnings.warn(message, RuntimeWarning, stacklevel=4)
            else:
                raise GuaranteeViolation(
                    message + " Pass allow_weaker=True to proceed anyway."
                )
        return engine
