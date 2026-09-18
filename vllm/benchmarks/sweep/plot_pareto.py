# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot the Pareto frontier between tokens/s/user and tokens/s/GPU from
saved parameter-sweep results. Derives per-user and per-GPU throughput for
every summary.json record, keeps the non-dominated points, and saves one
annotated scatter plot under <experiment_dir>/pareto. Registered as the
`vllm bench sweep plot_pareto` subcommand (SweepPlotParetoArgs.parser_name,
cli.py SUBCOMMANDS).
"""
import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.collection_utils import full_groupby
from vllm.utils.import_utils import PlaceholderModule

from .plot import DummyExecutor, _json_load_bytes
from .utils import sanitize_filename

if TYPE_CHECKING:
    import pandas as pd


def _first_present(run_data: dict[str, object], keys: list[str]):
    """Return the first value found in run_data across a list of candidate
    key names, trying each key in its original form and with '_'/'-' separators
    swapped. Returns None when no candidate key is present in the record.
    """
    for key in keys:
        for candidate in {key, key.replace("_", "-"), key.replace("-", "_")}:
            if candidate in run_data:
                return run_data[candidate]
    return None


def _get_numeric(
    run_data: dict[str, object],
    keys: list[str],
    *,
    allow_zero: bool = True,
) -> float | None:
    """Resolve a numeric value from run_data by candidate keys (via
    _first_present) and convert it to float. Raises ValueError naming the
    offending value when conversion fails. Returns None when no candidate
    key is present, or when the value is 0 and allow_zero is False.
    """
    value = _first_present(run_data, keys)
    if value is None:
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected numeric value for one of {keys}, "
            f"but found {value!r} in {run_data=}"
        ) from exc

    if not allow_zero and numeric == 0:
        return None

    return numeric


def _infer_user_count(
    run_data: dict[str, object],
    user_count_var: str | None,
) -> float | None:
    """Estimate the concurrent user count for one sweep run. Tries the configured
    user_count_var then the literal key 'request_rate' (non-zero values only, via
    _get_numeric allow_zero=False); when neither yields a value, falls back to the
    observed 'max_concurrent_requests'. Returns None when no candidate is usable,
    which makes _prepare_records skip the run.
    """
    candidates = [user_count_var] if user_count_var else []
    candidates.extend(["request_rate"])
    user_count = _get_numeric(run_data, candidates, allow_zero=False)
    if user_count is not None:
        return user_count

    # Fallback to the observed peak if configured value is missing.
    return _get_numeric(run_data, ["max_concurrent_requests"], allow_zero=False)


def _infer_gpu_count(
    run_data: dict[str, object],
    gpu_count_var: str | None,
) -> float:
    """Estimate the GPU count for one sweep run. Uses the configured gpu_count_var
when present (non-zero); otherwise multiplies tensor_parallel_size,
pipeline_parallel_size, and data_parallel_size (each also matched with
'-'/'_' key variants), starting from 1.0. A run record with none of
those keys therefore reports 1 GPU.
"""
    direct_candidates = [gpu_count_var] if gpu_count_var else []
    direct_gpu_count = _get_numeric(run_data, direct_candidates, allow_zero=False)
    if direct_gpu_count:
        return direct_gpu_count

    tp_size = _get_numeric(run_data, ["tensor_parallel_size", "tp"])
    pp_size = _get_numeric(run_data, ["pipeline_parallel_size", "pp"])
    dp_size = _get_numeric(run_data, ["data_parallel_size", "dp"])
    world_size = 1.0
    if tp_size:
        world_size *= tp_size
    if pp_size:
        world_size *= pp_size
    if dp_size:
        world_size *= dp_size

    return world_size


def _get_throughput(
    run_data: dict[str, object],
    throughput_var: str,
) -> float:
    """Return the run record's output throughput under the given key, raising
    ValueError listing the record's available keys when the key is absent or
    non-numeric.
    """
    throughput = _get_numeric(run_data, [throughput_var])
    if throughput is None:
        raise ValueError(
            f"Cannot find throughput metric {throughput_var!r} in run data. "
            f"Available keys: {sorted(run_data)}"
        )

    return throughput


def _prepare_records(
    all_data: list[dict[str, object]],
    *,
    user_count_var: str | None,
    gpu_count_var: str | None,
) -> tuple[list[dict[str, object]], int]:
    """Expand raw sweep records into plot-ready records. For each record
    it reads output_throughput (fatal when missing), estimates the user count
    (the record is skipped and counted when that fails) and the GPU count,
    then adds tokens_per_user, tokens_per_gpu, user_count_estimate, and
    gpu_count fields on top of the original record. Returns (prepared_records,
    skipped_missing_users_count).
    """
    prepared = []
    skipped_missing_users = 0

    for record in all_data:
        throughput = _get_throughput(record, "output_throughput")
        user_count = _infer_user_count(record, user_count_var)
        if user_count is None:
            skipped_missing_users += 1
            continue

        gpu_count = _infer_gpu_count(record, gpu_count_var)
        tokens_per_user = throughput / user_count
        tokens_per_gpu = throughput / gpu_count

        prepared.append(
            {
                **record,
                "tokens_per_user": tokens_per_user,
                "tokens_per_gpu": tokens_per_gpu,
                "user_count_estimate": user_count,
                "gpu_count": gpu_count,
            }
        )

    return prepared, skipped_missing_users


def _pareto_frontier(
    df: "pd.DataFrame",
    x_col: str,
    y_col: str,
    *,
    epsilon: float = 1e-9,
) -> "pd.DataFrame":
    """Extract the non-dominated (Pareto) frontier of a 2-D point set where
    both axes are to be maximized. Sorts points by x descending, then y descending,
    and keeps a point whenever its y is within epsilon of the best y seen so far
    — x is already maximal among the remaining rows, so matching the running
    best y means no kept point dominates it. Returns the original DataFrame rows
    at the kept indices.
    """
    sorted_df = df.sort_values([x_col, y_col], ascending=[False, False])
    frontier_indices = []
    best_y = -math.inf

    for idx, row in sorted_df.iterrows():
        y_val = row[y_col]
        if y_val >= best_y - epsilon:
            frontier_indices.append(idx)
            best_y = max(best_y, y_val)

    # Keep the row when its y is within epsilon of the best y so far: x
    # is already non-increasing (sorted descending), so a row still matching
    # the running best y is not dominated by any previously kept row;
    # epsilon admits near-duplicate y values as frontier points.
    return df.loc[frontier_indices]


def _get_fig_path(
    fig_dir: Path,
    fig_group: tuple[tuple[str, str], ...],
) -> Path:
    """Build the output path for the Pareto figure: <fig_dir>/PARETO.png,
    with group key=value segments appended when a group is present, the whole
    name passed through sanitize_filename.
    """
    parts = ["PARETO"]
    if fig_group:
        parts.extend(f"{k}={v}" for k, v in fig_group)
    filename = sanitize_filename("-".join(parts) + ".png")
    return fig_dir / filename


def _plot_fig(
    fig_dir: Path,
    fig_group_data: tuple[tuple[tuple[str, str], ...], list[dict[str, object]]],
    label_by: list[str],
    *,
    dry_run: bool,
):
    """Draw the Pareto frontier for one group and save it under fig_dir.
    Converts the group records to a DataFrame, drops rows with missing per-user
    or per-GPU throughput, prints the planned output path, and returns early
    in dry_run without importing matplotlib. Computes the frontier, renders
    all points as a grey scatter with the frontier overlaid as a line,
    annotates each frontier point with the requested key=value pairs when
    label_by is given, and saves the figure.
    """
    fig_group, fig_data = fig_group_data
    fig_path = _get_fig_path(fig_dir, fig_group)

    print("[BEGIN FIGURE]")
    print(f"Group: {dict(fig_group)}")
    print(f"Output file: {fig_path}")

    if dry_run:
        print("[END FIGURE]")
        return

    # Lazy-import matplotlib/pandas/seaborn
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = PlaceholderModule("matplotlib").placeholder_attr("pyplot")
    try:
        import pandas as pd
    except ImportError:
        pd = PlaceholderModule("pandas")
    try:
        import seaborn as sns
    except ImportError:
        sns = PlaceholderModule("seaborn")

    df = pd.DataFrame.from_records(fig_data)
    df = df.dropna(subset=["tokens_per_user", "tokens_per_gpu"])

    if df.empty:
        print("No data points available after filtering; skipping.")
        print("[END FIGURE]")
        return

    frontier = _pareto_frontier(df, "tokens_per_user", "tokens_per_gpu")
    frontier = frontier.sort_values("tokens_per_user")

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="tokens_per_user",
        y="tokens_per_gpu",
        color="0.5",
        alpha=0.6,
        ax=ax,
        label="All runs",
    )
    sns.lineplot(
        data=frontier,
        x="tokens_per_user",
        y="tokens_per_gpu",
        marker="o",
        ax=ax,
        label="Pareto frontier",
    )

    if label_by:
        for _, row in frontier.iterrows():
            label_parts = []
            for key in label_by:
                if key in row:
                    label_parts.append(f"{key}={row[key]}")
            if label_parts:
                ax.text(
                    row["tokens_per_user"],
                    row["tokens_per_gpu"],
                    "\n".join(label_parts),
                    fontsize=8,
                )

    ax.set_xlabel("Tokens/s/user")
    ax.set_ylabel("Tokens/s/GPU")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)

    print(
        f"Plotted {len(df)} points; Pareto frontier size: {len(frontier)}.",
    )
    print("[END FIGURE]")


def plot_pareto(
    output_dir: Path,
    user_count_var: str | None,
    gpu_count_var: str | None,
    label_by: list[str],
    *,
    dry_run: bool,
):
    """Entry point for the plot_pareto subcommand. Loads every summary.json
    under output_dir (raising ValueError when none are found), prepares the
    records via _prepare_records (printing how many runs were skipped for a
    missing user count), raises ValueError when no record survives, and
    dispatches _plot_fig over the resulting groups into <output_dir>/pareto
    via the executor (in-process DummyExecutor when there is a single group).
    """
    fig_dir = output_dir / "pareto"
    raw_data = [
        run_data
        for path in output_dir.rglob("**/summary.json")
        for run_data in _json_load_bytes(path)
    ]

    if not raw_data:
        raise ValueError(f"Did not find any parameter sweep results under {output_dir}")

    fig_dir.mkdir(parents=True, exist_ok=True)

    prepared_data, skipped_missing_users = _prepare_records(
        raw_data,
        user_count_var=user_count_var,
        gpu_count_var=gpu_count_var,
    )

    if skipped_missing_users:
        print(
            f"Skipped {skipped_missing_users} runs without a user count "
            "(`max_concurrency` or `max_concurrent_requests`).",
        )

    if not prepared_data:
        raise ValueError(
            "No data points with both throughput and user count available "
            "to plot Pareto frontier.",
        )

    fig_groups = full_groupby(
        prepared_data,
        key=lambda item: tuple(),
    )

    with DummyExecutor() if len(fig_groups) <= 1 else ProcessPoolExecutor() as executor:
        all(
            executor.map(
                partial(
                    _plot_fig,
                    fig_dir,
                    label_by=label_by,
                    dry_run=dry_run,
                ),
                fig_groups,
            )
        )


@dataclass
class SweepPlotParetoArgs:
    """CLI argument holder for the `vllm bench sweep plot_pareto` subcommand.
    from_cli_args() builds the instance from the parsed Namespace (splitting
    --label-by); add_cli_args() registers the subparser arguments;
    run_main() forwards the fields to plot_pareto().
    """
    output_dir: Path
    user_count_var: str | None
    gpu_count_var: str | None
    label_by: list[str]
    dry_run: bool

    parser_name: ClassVar[str] = "plot_pareto"
    parser_help: ClassVar[str] = (
        "Plot Pareto frontier between tokens/s/user and tokens/s/GPU "
        "from parameter sweep results."
    )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        """Build a SweepPlotParetoArgs from parsed CLI args. Verifies EXPERIMENT_DIR
        exists (raising ValueError otherwise) and splits the --label-by comma
        list (None or empty string -> []).
        """
        output_dir = Path(args.EXPERIMENT_DIR)
        if not output_dir.exists():
            raise ValueError(f"No parameter sweep results under {output_dir}")

        label_by = [] if not args.label_by else args.label_by.split(",")

        return cls(
            output_dir=output_dir,
            user_count_var=args.user_count_var,
            gpu_count_var=args.gpu_count_var,
            label_by=label_by,
            dry_run=args.dry_run,
        )

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser):
        """Register the plot_pareto subcommand arguments: EXPERIMENT_DIR positional,
        --user-count-var (default 'max_concurrency'), --gpu-count-var (default None),
        --label-by (default 'max_concurrency,gpu_count'), and --dry-run. Returns the
        parser.
        """
        parser.add_argument(
            "EXPERIMENT_DIR",
            type=str,
            help="The directory containing the sweep results to plot.",
        )
        parser.add_argument(
            "--user-count-var",
            type=str,
            default="max_concurrency",
            help="Result key that stores concurrent user count. "
            "Falls back to max_concurrent_requests if missing.",
        )
        parser.add_argument(
            "--gpu-count-var",
            type=str,
            default=None,
            help="Result key that stores GPU count. "
            "If not provided, falls back to num_gpus/gpu_count "
            "or tensor_parallel_size * pipeline_parallel_size.",
        )
        parser.add_argument(
            "--label-by",
            type=str,
            default="max_concurrency,gpu_count",
            help="Comma-separated list of fields to annotate on Pareto frontier "
            "points.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="If set, prints the figures to plot without drawing them.",
        )

        return parser


def run_main(args: SweepPlotParetoArgs):
    """Forward the SweepPlotParetoArgs fields to plot_pareto() and return
    its result.
    """
    return plot_pareto(
        output_dir=args.output_dir,
        user_count_var=args.user_count_var,
        gpu_count_var=args.gpu_count_var,
        label_by=args.label_by,
        dry_run=args.dry_run,
    )


def main(args: argparse.Namespace):
    """CLI entry point for the plot_pareto subcommand: build SweepPlotParetoArgs
    from the parsed Namespace and call run_main.
    """
    run_main(SweepPlotParetoArgs.from_cli_args(args))


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=SweepPlotParetoArgs.parser_help)
    SweepPlotParetoArgs.add_cli_args(parser)

    main(parser.parse_args())
