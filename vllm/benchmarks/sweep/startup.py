# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep subcommand for startup-time benchmarks: expands the serve_params
x startup_params Cartesian product, rewrites each `vllm bench startup`
invocation with the combination overrides and a forced --output-json,
caches per-run JSON under experiment_dir (skipping runs whose output
already exists), and merges every record into summary.csv."""

import argparse
import json
import shlex
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

from vllm.benchmarks.startup import add_cli_args as add_startup_cli_args
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.import_utils import PlaceholderModule

from .param_sweep import ParameterSweep, ParameterSweepItem
from .utils import sanitize_filename

try:
    import pandas as pd
except ImportError:
    pd = PlaceholderModule("pandas")


@lru_cache(maxsize=1)
def _get_supported_startup_keys() -> set[str]:
    """lru_cached set of every key the real `vllm bench startup` CLI accepts:
    each argparse dest plus each --option string normalized to underscore form.
    Built once by parsing the genuine startup parser with add_help=False."""
    parser = FlexibleArgumentParser(add_help=False)
    add_startup_cli_args(parser)

    supported: set[str] = {"config"}
    for action in parser._actions:
        if action.dest and action.dest is not argparse.SUPPRESS:
            supported.add(action.dest)
        for option in action.option_strings:
            if option.startswith("--"):
                supported.add(option.lstrip("-").replace("-", "_"))

    return supported


def _is_supported_param(param_key: str, supported: set[str]) -> bool:
    """Decides whether a sweep key may be applied: _benchmark_name always
    passes; otherwise the first dotted segment of the key is dash-normalized and
    checked against the supported set."""
    if param_key == "_benchmark_name":
        return True
    prefix = param_key.split(".", 1)[0]
    normalized = prefix.replace("-", "_")
    return normalized in supported


def _filter_params(
    params: ParameterSweep, *, supported: set[str], strict: bool
) -> ParameterSweep:
    """Projects every ParameterSweepItem onto the supported keys. Dropped
    keys are reported per item (label from _benchmark_name or as_text); strict=True
    raises ValueError, otherwise the message is printed. A fully dropped item becomes
    an empty combination."""
    filtered = []
    for item in params:
        kept: dict[str, object] = {}
        dropped: list[str] = []
        for key, value in item.items():
            if _is_supported_param(key, supported):
                kept[key] = value
            else:
                dropped.append(key)

        if dropped:
            label = item.get("_benchmark_name") or item.as_text()
            message = (
                "Ignoring unsupported startup params"
                f"{' for ' + str(label) if label else ''}: "
                f"{', '.join(sorted(dropped))}"
            )
            if strict:
                raise ValueError(message)
            print(message)

        filtered.append(ParameterSweepItem.from_record(kept))

    return ParameterSweep(filtered)


def _update_run_data(
    run_data: dict[str, object],
    serve_overrides: ParameterSweepItem,
    startup_overrides: ParameterSweepItem,
    run_number: int,
) -> dict[str, object]:
    """Stamps run_number onto the benchmark JSON record and overlays the
    serve combination keys, then the startup combination keys (startup wins on key
    conflicts)."""
    run_data["run_number"] = run_number
    run_data.update(serve_overrides)
    run_data.update(startup_overrides)
    return run_data


def _strip_arg(cmd: list[str], keys: tuple[str, ...]) -> list[str]:
    """Removes flags from a command list in both forms: the space form (the
    flag plus its following value) and the --flag=value form."""
    stripped: list[str] = []
    skip_next = False
    for arg in cmd:
        if skip_next:
            skip_next = False
            continue
        if arg in keys:
            skip_next = True
            continue
        if any(arg.startswith(f"{key}=") for key in keys):
            continue
        stripped.append(arg)
    return stripped


def _apply_output_json(cmd: list[str], output_path: Path) -> list[str]:
    """Strips any existing --output-json / --output_json from the command
    and appends the sweep's own, pointing at the per-run output path, so the result
    location never depends on the user's startup_cmd."""
    keys = ("--output-json", "--output_json")
    cmd = _strip_arg(cmd, keys)
    return [*cmd, keys[0], str(output_path)]


def _get_comb_base_path(
    experiment_dir: Path,
    serve_comb: ParameterSweepItem,
    startup_comb: ParameterSweepItem,
) -> Path:
    """Names the per-combination dir: SERVE-<name> and/or STARTUP-<name>
    segments (whichever combination is non-empty) joined by '-' and passed through
    sanitize_filename."""
    parts = list[str]()
    if serve_comb:
        parts.extend(("SERVE-", serve_comb.name))
    if startup_comb:
        parts.extend(("STARTUP-", startup_comb.name))

    return experiment_dir / sanitize_filename("-".join(parts))


def _get_comb_run_path(base_path: Path, run_number: int | None) -> Path:
    """run_number=None maps to the combination's summary.json; a concrete
    run number maps to run=<n>.json."""
    if run_number is None:
        return base_path / "summary.json"
    return base_path / f"run={run_number}.json"


def run_benchmark(
    startup_cmd: list[str],
    *,
    serve_overrides: ParameterSweepItem,
    startup_overrides: ParameterSweepItem,
    run_number: int,
    output_path: Path,
    show_stdout: bool,
    dry_run: bool,
) -> dict[str, object] | None:
    """Executes one run of one combination: applies serve then startup overrides
    to startup_cmd, forces --output-json, and if the output file already exists
    reuses it (resume path). Otherwise it runs the command with check=True, reads
    the JSON back, overlays the combination keys, and rewrites the file. dry_run
    prints the plan and returns None."""
    cmd = serve_overrides.apply_to_cmd(startup_cmd)
    cmd = startup_overrides.apply_to_cmd(cmd)
    cmd = _apply_output_json(cmd, output_path)

    print("[BEGIN BENCHMARK]")
    print(f"Serve overrides: {serve_overrides}")
    print(f"Startup overrides: {startup_overrides}")
    print(f"Run Number: {run_number}")
    print(f"Benchmark command: {cmd}")
    print(f"Output file: {output_path}")

    if output_path.exists():
        print("Found existing results.")
        print("[SKIPPED BENCHMARK]")

        with output_path.open("r", encoding="utf-8") as f:
            run_data = json.load(f)
            return _update_run_data(
                run_data, serve_overrides, startup_overrides, run_number
            )

    if dry_run:
        print("[END BENCHMARK]")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        cmd,
        stdout=None if show_stdout else subprocess.DEVNULL,
        check=True,
    )

    with output_path.open("r", encoding="utf-8") as f:
        run_data = json.load(f)

    run_data = _update_run_data(
        run_data, serve_overrides, startup_overrides, run_number
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=4)

    print("[END BENCHMARK]")
    return run_data


def run_comb(
    startup_cmd: list[str],
    *,
    serve_comb: ParameterSweepItem,
    startup_comb: ParameterSweepItem,
    base_path: Path,
    num_runs: int,
    show_stdout: bool,
    dry_run: bool,
) -> list[dict[str, object]] | None:
    """Runs num_runs of one combination into run=<n>.json files, collects
    the per-run records, and writes the combination's summary.json; returns None
    in dry-run mode before writing."""
    comb_data = list[dict[str, object]]()
    for run_number in range(num_runs):
        run_data = run_benchmark(
            startup_cmd,
            serve_overrides=serve_comb,
            startup_overrides=startup_comb,
            run_number=run_number,
            output_path=_get_comb_run_path(base_path, run_number),
            show_stdout=show_stdout,
            dry_run=dry_run,
        )
        if run_data is not None:
            comb_data.append(run_data)

    if dry_run:
        return None

    with _get_comb_run_path(base_path, run_number=None).open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(comb_data, f, indent=4)

    return comb_data


def run_combs(
    startup_cmd: list[str],
    *,
    serve_params: ParameterSweep,
    startup_params: ParameterSweep,
    experiment_dir: Path,
    num_runs: int,
    show_stdout: bool,
    dry_run: bool,
) -> "pd.DataFrame | None":
    """Cartesian product of serve_params and startup_params through run_comb;
    in real mode merges every record into a pandas DataFrame and
    writes experiment_dir/summary.csv.
    Returns None in dry-run mode."""
    all_data = list[dict[str, object]]()
    for serve_comb in serve_params:
        for startup_comb in startup_params:
            base_path = _get_comb_base_path(experiment_dir, serve_comb, startup_comb)
            comb_data = run_comb(
                startup_cmd,
                serve_comb=serve_comb,
                startup_comb=startup_comb,
                base_path=base_path,
                num_runs=num_runs,
                show_stdout=show_stdout,
                dry_run=dry_run,
            )
            if comb_data is not None:
                all_data.extend(comb_data)

    if dry_run:
        return None

    combined_df = pd.DataFrame.from_records(all_data)
    combined_df.to_csv(experiment_dir / "summary.csv")
    return combined_df


@dataclass
class SweepStartupArgs:
    """Argument dataclass for the `vllm bench sweep startup` subcommand.
    Parses the startup command string, loads the two sweep JSON files (defaulting
    each to a single empty combination), filters them to startup-supported keys
    (strict optional), and names the experiment by argument or timestamp."""
    startup_cmd: list[str]
    serve_params: ParameterSweep
    startup_params: ParameterSweep
    output_dir: Path
    experiment_name: str
    num_runs: int
    show_stdout: bool
    dry_run: bool
    resume: bool

    parser_name: ClassVar[str] = "startup"
    parser_help: ClassVar[str] = (
        "Benchmark vLLM startup time over parameter combinations."
    )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        """shlex.splits the startup command, loads --serve-params / --startup-params
        (a single empty combination when omitted), filters both
        against _get_supported_startup_keys
        under --strict-params, defaults the experiment name to a timestamp, and requires
        num_runs >= 1."""
        startup_cmd = shlex.split(args.startup_cmd)

        if args.serve_params:
            serve_params = ParameterSweep.read_json(args.serve_params)
        else:
            serve_params = ParameterSweep.from_records([{}])

        if args.startup_params:
            startup_params = ParameterSweep.read_json(args.startup_params)
        else:
            startup_params = ParameterSweep.from_records([{}])

        supported = _get_supported_startup_keys()
        strict_params = args.strict_params
        serve_params = _filter_params(
            serve_params, supported=supported, strict=strict_params
        )
        startup_params = _filter_params(
            startup_params, supported=supported, strict=strict_params
        )

        if args.experiment_name:
            experiment_name = args.experiment_name
        else:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        if args.num_runs < 1:
            raise ValueError("`num_runs` should be at least 1.")

        return cls(
            startup_cmd=startup_cmd,
            serve_params=serve_params,
            startup_params=startup_params,
            output_dir=Path(args.output_dir),
            experiment_name=experiment_name,
            num_runs=args.num_runs,
            show_stdout=args.show_stdout,
            dry_run=args.dry_run,
            resume=args.resume,
        )

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Registers the subcommand flags: --startup-cmd (default 'vllm bench
        startup'), --serve-params, --startup-params, --strict-params, -o/--output-dir,
        -e/--experiment-name, --num-runs, --show-stdout, --dry-run, --resume."""
        parser.add_argument(
            "--startup-cmd",
            type=str,
            default="vllm bench startup",
            help="The command used to run the startup benchmark.",
        )

        parser.add_argument(
            "--serve-params",
            type=str,
            default=None,
            help="Path to JSON file containing parameter combinations "
            "for the `vllm serve` command. Only parameters supported by "
            "`vllm bench startup` will be applied.",
        )
        parser.add_argument(
            "--startup-params",
            type=str,
            default=None,
            help="Path to JSON file containing parameter combinations "
            "for the `vllm bench startup` command.",
        )
        parser.add_argument(
            "--strict-params",
            action="store_true",
            help="If set, unknown parameters in sweep files raise an error "
            "instead of being ignored.",
        )

        parser.add_argument(
            "-o",
            "--output-dir",
            type=str,
            default="results",
            help="The main directory to which results are written.",
        )
        parser.add_argument(
            "-e",
            "--experiment-name",
            type=str,
            default=None,
            help="The name of this experiment (defaults to current timestamp). "
            "Results will be stored under `output_dir/experiment_name`.",
        )
        parser.add_argument(
            "--num-runs",
            type=int,
            default=1,
            help="Number of runs per parameter combination.",
        )
        parser.add_argument(
            "--show-stdout",
            action="store_true",
            help="If set, logs the standard output of subcommands.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="If set, prints the commands to run, "
            "then exits without executing them.",
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            help="Resume a previous execution of this script, i.e., only run "
            "parameter combinations for which there are still no output files "
            "under `output_dir/experiment_name`.",
        )

        return parser

    def resolve_experiment_dir(self) -> Path:
        """Computes output_dir/experiment_name and enforces the resume contract:
        --resume requires the dir to exist, a fresh run requires it to be absent; either
        violation raises ValueError."""
        experiment_dir = self.output_dir / self.experiment_name

        if self.resume:
            if not experiment_dir.exists():
                raise ValueError(f"Cannot resume from non-existent {experiment_dir=}")
        else:
            if experiment_dir.exists():
                raise ValueError(f"Cannot overwrite existing {experiment_dir=}")

        return experiment_dir

    @contextmanager
    def run_ctx(self, experiment_dir: Path):
        """Context manager around the whole experiment: in dry-run it yields
        and prints the save path; in real mode it re-wraps any
        BaseException in a RuntimeError
        pointing at --resume, so a partially finished experiment can be continued."""
        if self.dry_run:
            yield
            print(f"Experiment will be saved at: {experiment_dir}")
            return

        try:
            yield
            print(f"Experiment has been saved at: {experiment_dir}")
        except BaseException as exc:
            raise RuntimeError(
                "The script was terminated early. Use `--resume` "
                "to continue the script from its last checkpoint."
            ) from exc


def run_main(args: SweepStartupArgs):
    """Resolves the experiment dir and, under run_ctx, runs run_combs with
    the parsed fields."""
    experiment_dir = args.resolve_experiment_dir()

    with args.run_ctx(experiment_dir):
        return run_combs(
            startup_cmd=args.startup_cmd,
            serve_params=args.serve_params,
            startup_params=args.startup_params,
            experiment_dir=experiment_dir,
            num_runs=args.num_runs,
            show_stdout=args.show_stdout,
            dry_run=args.dry_run,
        )


def main(args: argparse.Namespace):
    """CLI entry point: converts the parsed argparse namespace into SweepStartupArgs
    and calls run_main."""
    run_main(SweepStartupArgs.from_cli_args(args))


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=SweepStartupArgs.parser_help)
    SweepStartupArgs.add_cli_args(parser)
    main(parser.parse_args())
