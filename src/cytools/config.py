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
# Description:  This module contains various configuration variables for
#               experimental features and custom installations.
# -----------------------------------------------------------------------------

# 'standard' imports
import os
import warnings

# The number of CPU threads to use in some computations, such as finding the
# extremal rays of a cone. When set to None, then it uses all available threads.
n_threads = None

# Mosek license
# Default: defer to Mosek's own license discovery (the MOSEKLM_LICENSE_FILE
# environment variable, or ~/mosek/mosek.lic). set_mosek_path() overrides this.
_mosek_license = None
_mosek_is_activated = None
_mosek_error = ""


def check_mosek_license(silent=False):
    """
    **Description:**
    Checks if the Mosek license is valid. If it is not, it prints the reason.

    **Arguments:**
    None.

    **Returns:**
    Nothing.

    **Example:**
    The Mosek license should be automatically checked, but it can also be
    checked as follows.
    ```python {2}
    import cytools
    cytools.config.check_mosek_license()
    # It will print an error if it is not working, and if nothing is printed
    # then it is working correctly
    ```
    """
    global _mosek_license
    if _mosek_license is not None:
        os.environ["MOSEKLM_LICENSE_FILE"] = _mosek_license
    global _mosek_error
    global _mosek_is_activated
    # The import is in its own try. When it was inside the block below, an
    # import failure that was *not* an ImportError -- an OSError from an
    # unloadable libmosek shared object is the realistic one -- fell through to
    # `except mosek.Error`, and evaluating that clause raised
    # `UnboundLocalError: cannot access local variable 'mosek'`, masking the
    # real error and escaping the `except Exception` fallback entirely.
    try:
        import mosek
    except ImportError:
        _mosek_error = "Info: Mosek is not installed."
        _mosek_is_activated = False
        if not silent:
            print(_mosek_error)
        return
    except Exception as e:
        _mosek_error = (
            "Info: Mosek is installed but could not be imported "
            f"({type(e).__name__}: {e}). An alternative optimizer will be used."
        )
        _mosek_is_activated = False
        if not silent:
            print(_mosek_error)
        return

    try:
        mosek.Env().Task(0, 0).optimize()
        _mosek_is_activated = True
        if not silent:
            print("Mosek was successfully activated.")
    except mosek.Error as e:
        _mosek_error = (
            "Info: Mosek is not activated. "
            "An alternative optimizer will be used.\n"
            f"Error encountered: {e}"
        )
        _mosek_is_activated = False
    except Exception:
        _mosek_error = (
            "Info: There was a problem with Mosek. "
            "An alternative optimizer will be used."
        )
        _mosek_is_activated = False
    if not silent:
        print(_mosek_error)


def mosek_is_activated():
    global _mosek_error
    global _mosek_is_activated
    if _mosek_is_activated is None:
        check_mosek_license(silent=True)
    return _mosek_is_activated


def set_mosek_path(path):
    """
    **Description:**
    Sets a custom path to the Mosek license, for when it is stored in a
    non-default location on your computer. The license will be checked after
    the new path is set.

    **Arguments:**
    - `path` *(str)*: The path to the Mosek license.

    **Returns:**
    Nothing.

    **Example:**
    ```python {2}
    import cytools
    cytools.config.set_mosek_path("/path/to/mosek.lic")
    ```
    """
    global _mosek_license
    _mosek_license = path
    check_mosek_license()


def engines(*, allow_weaker: bool = False, **choices: str):
    """
    **Description:**
    Force specific computational engines for the duration of a `with` block.

    CYTools normally selects engines itself. A call site declares the
    mathematical guarantees it depends on -- exact arithmetic, a certified
    infeasibility answer, a regular-by-construction triangulation -- and the
    cheapest available engine providing them is used. That decision is not a
    user preference: `Cone.is_solid` reads a missing interior point as "the
    cone is not full-dimensional", so an optimizer that cannot distinguish
    "infeasible" from "I gave up" returns a *different answer* there, not a
    slower one.

    This function exists for the cases where the choice really is the user's:
    reproducing a published run bit for bit, cross-checking two independent
    implementations against each other, and bisecting a numerical
    disagreement.

    An engine that cannot provide what a call site requires raises
    `GuaranteeViolation` rather than silently returning a weaker result. Pass
    `allow_weaker=True` to downgrade that to a warning, which is what a
    differential test comparing a strong engine against a weak one needs.

    :::note
    The setting is scoped to the current context, so it does not leak into
    other threads and does not cross a process boundary. Worker processes
    resolve engines from their own capabilities. A pool that must inherit the
    parent's choices should pass `cytools.config.engine_overrides()` to its
    workers and apply it there with `cytools.config.set_engine_overrides()`.
    :::

    **Arguments:**
    - `allow_weaker`: Permit an engine whose guarantees are weaker than the
        call site requires.
    - `**choices`: Task name to engine name. Task names are `convex_hull`,
        `interior_point`, `stretched_tip`, `triangulate` and `linear_solve`.

    **Returns:**
    A context manager.

    **Example:**
    Check that the exact and floating-point convex hulls agree on a polytope.
    ```python {4}
    import cytools
    p = cytools.Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
    exact = p.inequalities()
    with cytools.config.engines(convex_hull="qhull", allow_weaker=True):
        approx = cytools.Polytope(p.points()).inequalities()
    ```
    """
    from cytools._backends.registry import override

    return override(allow_weaker=allow_weaker, **choices)


def available_engines() -> dict[str, tuple[str, ...]]:
    """
    **Description:**
    The engines usable in this process, per task, in stable registration order.

    Useful for reporting what a given machine actually ran with: an engine
    absent here was never a candidate, whatever the documentation says.

    **Returns:**
    A mapping of task name to the available engine names.

    **Example:**
    ```python {2}
    import cytools
    cytools.config.available_engines()
    # {'convex_hull': ('interval', 'palp', 'ppl', 'qhull'), ...}
    ```
    """
    from cytools._backends.engines import all_registries

    return {r.task: r.available() for r in all_registries()}


def engine_overrides() -> dict[str, str]:
    """The engine overrides active in this context. See `engines`."""
    from cytools._backends.registry import get_overrides

    return get_overrides()


def set_engine_overrides(mapping) -> None:
    """Apply engine overrides in this process. See `engines`."""
    from cytools._backends.registry import set_overrides

    set_overrides(mapping)


# Lock experimental features by default.
_exp_features_enabled: bool = False


def enable_experimental_features():
    """
    **Description:**
    Enables the experimental features of CYTools. For more information read the
    [experimental features page](./experimental).

    **Arguments:**
    None.

    **Returns:**
    Nothing.

    **Example:**
    We enable the experimental features.
    ```python {2}
    import cytools
    cytools.config.enable_experimental_features()
    ```
    """
    global _exp_features_enabled
    _exp_features_enabled = True
    warnings.warn(
        "\n**************************************************************\n"
        "Warning: You have enabled experimental features of CYTools.\n"
        "Some of these features may be broken or not fully tested,\n"
        "and they may undergo significant changes in future versions.\n"
        "**************************************************************\n",
        UserWarning,
        stacklevel=2,
    )
