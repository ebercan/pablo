"""Single CLI for every featurization step.

    python -m pablo.featurization_and_kernel_scan.features --list
    python -m pablo.featurization_and_kernel_scan.features build map4 ecfp4
    python -m pablo.featurization_and_kernel_scan.features build --family fingerprint
    python -m pablo.featurization_and_kernel_scan.features submit unimol

``--list`` prints the registry, ``build`` runs steps locally, and ``submit``
sends GPU steps to Mechthild. ``build`` refuses a step
whose ``runtime`` does not match this machine unless ``--force`` is given, and
refuses one whose prerequisites have not produced their outputs unless
``--skip-prereq-check`` is given.
"""

from __future__ import annotations

import argparse
import shutil

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.features.registry import (
    FAMILIES,
    RUNTIMES,
    Featurizer,
    build_registry,
)


def _have_xtb() -> bool:
    """Whether the ``xtb`` binary is on PATH."""
    return shutil.which("xtb") is not None


def _outputs_present(entry: Featurizer) -> bool:
    """Whether every declared output of ``entry`` already exists on disk."""
    if not entry.outputs:
        return False
    return all((paths.FEAT_DIR / out).exists() or (paths.DATA_DIR / out).exists()
               for out in entry.outputs)


def _print_listing(registry: dict[str, Featurizer]) -> None:
    """Print the registry grouped by family, marking which outputs exist."""
    print(f"{len(registry)} featurizers  (* = outputs present)\n")
    for family in FAMILIES:
        entries = [e for e in registry.values() if e.family == family]
        if not entries:
            continue
        print(f"{family}:")
        for entry in entries:
            mark = "*" if _outputs_present(entry) else " "
            req = f"  requires: {', '.join(entry.requires)}" if entry.requires else ""
            print(f"  {mark} {entry.name:22s} [{entry.runtime:5s}] {entry.note}{req}")
        print()


def _select(registry: dict[str, Featurizer], names, family, runtime) -> list[Featurizer]:
    """Resolve the CLI selection into registry entries.

    Args:
        registry: The featurizer registry.
        names: Explicit featurizer names, possibly empty.
        family: Family filter, or ``None``.
        runtime: Runtime filter, or ``None``.

    Returns:
        The selected entries, in registry order.

    Raises:
        SystemExit: If a name is unknown or the selection is empty.
    """
    if names:
        unknown = [n for n in names if n not in registry]
        if unknown:
            raise SystemExit(
                f"Unknown featurizer(s): {', '.join(unknown)}. "
                "Run --list to see the registry."
            )
        selected = [registry[n] for n in names]
    else:
        selected = list(registry.values())

    if family:
        selected = [e for e in selected if e.family == family]
    if runtime:
        selected = [e for e in selected if e.runtime == runtime]
    if not selected:
        raise SystemExit("Selection matched no featurizers.")
    return selected


def _check_runnable(entry: Featurizer, registry, force: bool, skip_prereq: bool) -> None:
    """Refuse a step this machine or this state cannot support.

    Args:
        entry: The step about to run.
        registry: The full registry, for prerequisite lookup.
        force: Skip the runtime check.
        skip_prereq: Skip the prerequisite-output check.

    Raises:
        SystemExit: If the step cannot run and was not forced.
    """
    if not force:
        if entry.runtime == "xtb" and not _have_xtb():
            raise SystemExit(
                f"{entry.name}: needs the 'xtb' binary on PATH. Install it, or pass --force."
            )
        if entry.runtime == "gpu":
            raise SystemExit(
                f"{entry.name}: runtime 'gpu' — submit it to Mechthild with\n"
                f"  python -m pablo.featurization_and_kernel_scan.features submit {entry.name}\n"
                "or pass --force to run it here anyway."
            )
    if not skip_prereq:
        missing = [
            name for name in entry.requires
            if name in registry and not _outputs_present(registry[name])
        ]
        if missing:
            raise SystemExit(
                f"{entry.name}: prerequisite output missing for {', '.join(missing)}. "
                "Run those first, or pass --skip-prereq-check."
            )


def main() -> None:
    """Parse arguments and run the requested featurization steps."""
    parser = argparse.ArgumentParser(prog="features", description=__doc__)
    parser.add_argument("--list", action="store_true", help="List the registry and exit.")
    sub = parser.add_subparsers(dest="command")

    build = sub.add_parser("build", help="Run featurization steps.")
    build.add_argument("names", nargs="*", help="Featurizer names; empty means all.")
    build.add_argument("--family", choices=FAMILIES)
    build.add_argument("--runtime", choices=RUNTIMES)
    build.add_argument("--force", action="store_true",
                       help="Run despite a runtime mismatch.")
    build.add_argument("--skip-prereq-check", action="store_true",
                       help="Run without verifying prerequisite outputs.")
    build.add_argument("--dry-run", action="store_true",
                       help="Print what would run and exit.")
    build.add_argument("--cat-id", type=int, default=None,
                       help="Single-catalyst mode, for the Mechthild CPU array. "
                            "Only for steps that declare a per-catalyst builder.")

    submit = sub.add_parser("submit", help="Submit GPU featurizers to Mechthild.")
    submit.add_argument("names", nargs="*", help="Featurizer names; empty means all GPU ones.")

    args = parser.parse_args()
    registry = build_registry()

    if args.list or args.command is None:
        _print_listing(registry)
        return

    if args.command == "submit":
        from pablo.featurization_and_kernel_scan.features import foundation

        selected = _select(registry, args.names, None, "gpu")
        for entry in selected:
            print(f"[submit] {entry.name}")
            foundation.submit(entry.name)
        return

    selected = _select(registry, args.names, args.family, args.runtime)

    if args.cat_id is not None:
        unsupported = [e.name for e in selected if e.build_one is None]
        if unsupported:
            raise SystemExit(
                f"--cat-id is not supported by {', '.join(unsupported)}. "
                "Per-catalyst steps: "
                + ", ".join(e.name for e in registry.values() if e.build_one)
            )

    if args.dry_run:
        for entry in selected:
            print(f"would build {entry.name} [{entry.runtime}] -> {', '.join(entry.outputs) or '(cache only)'}")
        return

    for entry in selected:
        _check_runnable(entry, registry, args.force, args.skip_prereq_check)
    for entry in selected:
        suffix = f" cat_id={args.cat_id}" if args.cat_id is not None else ""
        print(f"\n=== {entry.name} [{entry.family}/{entry.runtime}]{suffix} ===")
        if args.cat_id is not None:
            entry.build_one(args.cat_id)
        else:
            entry.build()


if __name__ == "__main__":
    main()
