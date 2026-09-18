# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot performance curves from saved parameter-sweep results. Loads
every summary.json under an experiment directory, splits the records into one
seaborn figure per --fig-by group, applies --filter-by row predicates and --bin-by
column rounding, and saves one PNG per group. Registered as the `vllm bench
sweep plot` subcommand (SweepPlotArgs.parser_name, cli.py SUBCOMMANDS).
"""
import argparse
import json
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, ClassVar

from typing_extensions import Self, override

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.collection_utils import full_groupby
from vllm.utils.import_utils import PlaceholderModule

from .utils import sanitize_filename

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class PlotFilterBase(ABC):
    """Base class for the row-predicate classes registered in PLOT_FILTERS.
    Instances carry the column name (var) and the target value (target) parsed
    from one CLI filter token. Subclasses implement apply() to return the
    DataFrame rows satisfying the predicate. parse_str() parses a 'column OP
    value' token and picks the subclass from PLOT_FILTERS by operator.
    """
    var: str
    target: str

    @classmethod
    def parse_str(cls, s: str):
        """Parse one 'column OP value' filter token into a PlotFilterBase
        subclass instance. Operators are matched in PLOT_FILTERS insertion order
        (longest keys first: ==, !=, <=, >=, then < and >). The value keeps
        only the text after the operator with surrounding single/double quotes
        stripped. Raises ValueError naming the valid operators when no registered
        operator appears in the string.
        """
        for op_key in PLOT_FILTERS:
            if op_key in s:
                key, value = s.split(op_key)
                return PLOT_FILTERS[op_key](
                    key,
                    value.removeprefix(op_key).strip("'").strip('"'),
                )
        else:
            raise ValueError(
                f"Invalid operator for plot filter '{s}'. "
                f"Valid operators are: {sorted(PLOT_FILTERS)}",
            )

    @abstractmethod
    def apply(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Applies this filter to a DataFrame."""
        raise NotImplementedError


@dataclass
class PlotEqualTo(PlotFilterBase):
    """Row predicate that keeps rows whose column equals the target. The
    target is first converted to float so numeric-string values ('1000') compare
    equal against numeric columns; when float() fails the target is compared as
    a string.
    """
    @override
    def apply(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Filter the DataFrame to rows where df[var] equals target. Converts
        self.target to float when parseable (so numeric columns match numeric-string
        values) and falls back to the raw string otherwise.
        """
        target: float | str
        try:
            target = float(self.target)
        except ValueError:
            target = self.target

        return df[df[self.var] == target]


@dataclass
class PlotNotEqualTo(PlotFilterBase):
    """Row predicate that keeps rows whose column differs from the target.
    Same numeric-first coercion as PlotEqualTo: a parseable-numeric target
    compares as float, otherwise as string.
    """
    @override
    def apply(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Filter the DataFrame to rows where df[var] differs from target,
        with the target coerced to float when parseable.
        """
        target: float | str
        try:
            target = float(self.target)
        except ValueError:
            target = self.target

        return df[df[self.var] != target]


@dataclass
class PlotLessThan(PlotFilterBase):
    """Row predicate that keeps rows whose column is strictly less than the
    float target. The target must be numeric; float(self.target) raises a bare
    ValueError for non-numeric values.
    """
    @override
    def apply(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Filter the DataFrame to rows where df[var] < float(self.target).
        Unlike PlotEqualTo, a non-numeric target raises ValueError from
        float() without a message.
        """
        return df[df[self.var] < float(self.target)]


@dataclass
class PlotLessThanOrEqualTo(PlotFilterBase):
    """Row predicate that keeps rows whose column is less than or equal to
    the float target. The target must be numeric.
    """
    @override
    def apply(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Filter the DataFrame to rows where df[var] <= float(self.target).
        """
        return df[df[self.var] <= float(self.target)]


@dataclass
class PlotGreaterThan(PlotFilterBase):
    """Row predicate that keeps rows whose column is strictly greater than
    the float target. The target must be numeric.
    """
    @override
    def apply(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Filter the DataFrame to rows where df[var] > float(self.target).
        """
        return df[df[self.var] > float(self.target)]


@dataclass
class PlotGreaterThanOrEqualTo(PlotFilterBase):
    """Row predicate that keeps rows whose column is greater than or equal
    to the float target. The target must be numeric.
    """
    @override
    def apply(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Filter the DataFrame to rows where df[var] >= float(self.target).
        """
        return df[df[self.var] >= float(self.target)]


# NOTE: The ordering is important! Match longer op_keys first
PLOT_FILTERS: dict[str, type[PlotFilterBase]] = {
    "==": PlotEqualTo,
    "!=": PlotNotEqualTo,
    "<=": PlotLessThanOrEqualTo,
    ">=": PlotGreaterThanOrEqualTo,
    "<": PlotLessThan,
    ">": PlotGreaterThan,
}


class PlotFilters(list[PlotFilterBase]):
    """Ordered list of PlotFilterBase instances. parse_str() splits a
    comma-separated CLI string into one filter per item; apply() chains
    the predicates, so a row must survive every filter to remain in the
    result.
    """
    @classmethod
    def parse_str(cls, s: str):
        """Parse a comma-separated filter string into a PlotFilters
        list. An empty string yields an empty list (no filtering);
        each item is parsed by PlotFilterBase.parse_str.
        """
        if not s:
            return cls()

        return cls(PlotFilterBase.parse_str(e) for e in s.split(","))

    def apply(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Apply every predicate in order to the DataFrame and return
        the rows that pass all of them.
        """
        for item in self:
            df = item.apply(df)

        return df


@dataclass
class PlotBinner:
    """Rounds a column down to a multiple of bin_size so nearby points
    collapse into a single plotted value. parse_str() parses a 'column%bin_size'
    token from the PLOT_BINNERS registry.
    """
    var: str
    bin_size: float

    @classmethod
    def parse_str(cls, s: str):
        """Parse one 'column%bin_size' token into a PlotBinner. The operator
        is matched in PLOT_BINNERS insertion order and the value after '%'
        must parse as float. Raises ValueError when no registered operator
        appears.
        """
        for op_key in PLOT_BINNERS:
            if op_key in s:
                key, value = s.split(op_key)
                return PLOT_BINNERS[op_key](key, float(value.removeprefix(op_key)))
        else:
            raise ValueError(
                f"Invalid operator for plot binner '{s}'. "
                f"Valid operators are: {sorted(PLOT_BINNERS)}",
            )

    def apply(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Applies this binner to a DataFrame."""
        df = df.copy()
        df[self.var] = df[self.var] // self.bin_size * self.bin_size
        return df


# Registry mapping the binning operator '%' to PlotBinner.
# PlotBinner.parse_str() matches operators in this dict's insertion order,
# mirroring the PLOT_FILTERS registry above.
PLOT_BINNERS: dict[str, type[PlotBinner]] = {
    "%": PlotBinner,
}


class PlotBinners(list[PlotBinner]):
    """Ordered list of PlotBinner instances. parse_str() splits a comma-separated
    CLI string into one binner per item; apply() rounds each configured column
    in sequence on a copy of the DataFrame.
    """
    @classmethod
    def parse_str(cls, s: str):
        """Parse a comma-separated 'column%size' string into a PlotBinners
        list; an empty string yields no binning.
        """
        if not s:
            return cls()

        return cls(PlotBinner.parse_str(e) for e in s.split(","))

    def apply(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Apply each binner in sequence; each binner works on df.copy(), so
        the input frame is not mutated.
        """
        for item in self:
            df = item.apply(df)

        return df


def _json_load_bytes(path: Path) -> list[dict[str, object]]:
    with path.open("rb") as f:
        return json.load(f)


def _convert_inf_nan_strings(data: list[dict[str, object]]) -> list[dict[str, object]]:
    """
    Convert string values "inf", "-inf", and "nan" to their float equivalents.

    This handles the case where JSON serialization represents inf/nan as strings.
    """
    converted_data = []
    for record in data:
        converted_record: dict[str, object] = {}
        for key, value in record.items():
            if isinstance(value, str):
                if value in ["inf", "-inf", "nan"]:
                    converted_record[key] = float(value)
                else:
                    converted_record[key] = value
            else:
                converted_record[key] = value
        converted_data.append(converted_record)
    return converted_data


def _get_metric(run_data: dict[str, object], metric_key: str):
    try:
        return run_data[metric_key]
    except KeyError as exc:
        raise ValueError(f"Cannot find metric {metric_key!r} in {run_data=}") from exc


def _get_group(run_data: dict[str, object], group_keys: list[str]):
    return tuple((k, str(_get_metric(run_data, k))) for k in group_keys)


def _get_fig_path(fig_dir: Path, group: tuple[tuple[str, str], ...], fig_name: str):
    parts = list[str]()

    # Start with figure name (always provided, defaults to "FIGURE")
    parts.append(fig_name)

    # Always append group data if present
    if group:
        parts.extend(f"{k}={v}" for k, v in group)

    return fig_dir / sanitize_filename("-".join(parts) + ".png")


class DummyExecutor:
    """In-process stand-in for concurrent.futures.ProcessPoolExecutor. Exposes
    map as the builtin map and a no-op context manager, so plot() can dispatch
    _plot_fig through the same executor.map call without spawning worker
    processes for a single figure group.
    """
    map = map

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        return None


def _plot_fig(
    fig_dir: Path,
    fig_group_data: tuple[tuple[tuple[str, str], ...], list[dict[str, object]]],
    row_by: list[str],
    col_by: list[str],
    curve_by: list[str],
    *,
    var_x: str,
    var_y: str,
    filter_by: PlotFilters,
    bin_by: PlotBinners,
    scale_x: str | None,
    scale_y: str | None,
    dry_run: bool,
    fig_name: str,
    error_bars: bool,
    fig_height: float,
    fig_dpi: int,
):
    """Draw one figure for a single fig-by group and save it under fig_dir.
    Builds a row/column grid with full_groupby over row_by/col_by, checks that
    every requested variable exists in the result columns, applies PlotFilters
    then PlotBinners, and prints a skip note when no rows remain.
    With 3 or fewer curve_by variables they map one-to-one onto seaborn
    hue/style/size; beyond that the variables are concatenated into a single
    'curve_group' string used as hue. Saves the FacetGrid at fig_dpi
    and closes the figure.
    """
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

    fig_group, fig_data = fig_group_data

    row_groups = full_groupby(
        fig_data,
        key=lambda item: _get_group(item, row_by),
    )
    num_rows = len(row_groups)
    num_cols = max(
        len(full_groupby(row_data, key=lambda item: _get_group(item, col_by)))
        for _, row_data in row_groups
    )

    fig_path = _get_fig_path(fig_dir, fig_group, fig_name)

    print("[BEGIN FIGURE]")
    print(f"Group: {dict(fig_group)}")
    print(f"Grid: {num_rows} rows x {num_cols} cols")
    print(f"Output file: {fig_path}")

    if dry_run:
        print("[END FIGURE]")
        return

    # Convert string "inf", "-inf", and "nan" to their float equivalents
    fig_data = _convert_inf_nan_strings(fig_data)
    df = pd.DataFrame.from_records(fig_data)

    if var_x not in df.columns:
        raise ValueError(
            f"Cannot find {var_x=!r} in parameter sweep results. "
            f"Available variables: {df.columns.tolist()}"
        )
    if var_y not in df.columns:
        raise ValueError(
            f"Cannot find {var_y=!r} in parameter sweep results. "
            f"Available variables: {df.columns.tolist()}"
        )
    for k in row_by:
        if k not in df.columns:
            raise ValueError(
                f"Cannot find row_by={k!r} in parameter sweep results. "
                f"Available variables: {df.columns.tolist()}"
            )
    for k in col_by:
        if k not in df.columns:
            raise ValueError(
                f"Cannot find col_by={k!r} in parameter sweep results. "
                f"Available variables: {df.columns.tolist()}"
            )
    for k in curve_by:
        if k not in df.columns:
            raise ValueError(
                f"Cannot find curve_by={k!r} in parameter sweep results. "
                f"Available variables: {df.columns.tolist()}"
            )

    df = filter_by.apply(df)
    df = bin_by.apply(df)

    if len(df) == 0:
        print(f"No data to plot. Filters: {filter_by}")
        print("[END FIGURE]")
        return

    # Sort by curve_by columns alphabetically for consistent legend ordering
    if curve_by:
        df = df.sort_values(by=curve_by)

    df["row_group"] = (
        pd.concat(
            [k + "=" + df[k].astype(str) for k in row_by],
            axis=1,
        ).agg("\n".join, axis=1)
        if row_by
        else "(All)"
    )

    df["col_group"] = (
        pd.concat(
            [k + "=" + df[k].astype(str) for k in col_by],
            axis=1,
        ).agg("\n".join, axis=1)
        if col_by
        else "(All)"
    )

    if len(curve_by) <= 3:
        hue, style, size, *_ = (*curve_by, None, None, None)

        g = sns.relplot(
            df,
            x=var_x,
            y=var_y,
            hue=hue,
            style=style,
            size=size,
            markers=True,
            errorbar="sd" if error_bars else None,
            kind="line",
            row="row_group",
            col="col_group",
            height=fig_height,
        )
    else:
        df["curve_group"] = (
            pd.concat(
                [k + "=" + df[k].astype(str) for k in curve_by],
                axis=1,
            ).agg("\n".join, axis=1)
            if curve_by
            else "(All)"
        )

        g = sns.relplot(
            df,
            x=var_x,
            y=var_y,
            hue="curve_group",
            markers=True,
            errorbar="sd" if error_bars else None,
            kind="line",
            row="row_group",
            col="col_group",
            height=fig_height,
        )

    if row_by and col_by:
        g.set_titles("{row_name}\n{col_name}")
    elif row_by:
        g.set_titles("{row_name}")
    elif col_by:
        g.set_titles("{col_name}")
    else:
        g.set_titles("")

    if scale_x:
        g.set(xscale=scale_x)
    if scale_y:
        g.set(yscale=scale_y)

    g.savefig(fig_path, dpi=fig_dpi)
    plt.close(g.figure)

    print("[END FIGURE]")


def plot(
    output_dir: Path,
    fig_dir: Path,
    fig_by: list[str],
    row_by: list[str],
    col_by: list[str],
    curve_by: list[str],
    *,
    var_x: str,
    var_y: str,
    filter_by: PlotFilters,
    bin_by: PlotBinners,
    scale_x: str | None,
    scale_y: str | None,
    dry_run: bool,
    fig_name: str = "FIGURE",
    error_bars: bool = True,
    fig_height: float = 6.4,
    fig_dpi: int = 300,
):
    """Entry point for the plot subcommand. Loads every summary.json under
    output_dir (recursively), raises ValueError when none are found, groups
    the records by the fig_by variables, and dispatches _plot_fig per group
    — in a ProcessPoolExecutor when more than one group exists, otherwise
    in-process via DummyExecutor. all() forces the lazy executor.map
    generator to run.
    """
    all_data = [
        run_data
        for path in output_dir.rglob("**/summary.json")
        for run_data in _json_load_bytes(path)
    ]

    if not all_data:
        raise ValueError(f"Did not find any parameter sweep results under {output_dir}")

    fig_dir.mkdir(parents=True, exist_ok=True)

    fig_groups = full_groupby(
        all_data,
        key=lambda item: _get_group(item, fig_by),
    )

    with DummyExecutor() if len(fig_groups) <= 1 else ProcessPoolExecutor() as executor:
        # Resolve the iterable to ensure that the workers are run
        all(
            executor.map(
                partial(
                    _plot_fig,
                    fig_dir,
                    row_by=row_by,
                    col_by=col_by,
                    curve_by=curve_by,
                    var_x=var_x,
                    var_y=var_y,
                    filter_by=filter_by,
                    bin_by=bin_by,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    dry_run=dry_run,
                    fig_name=fig_name,
                    error_bars=error_bars,
                    fig_height=fig_height,
                    fig_dpi=fig_dpi,
                ),
                fig_groups,
            )
        )


@dataclass
class SweepPlotArgs:
    """CLI argument holder for the `vllm bench sweep plot` subcommand.
    from_cli_args() builds the instance from a parsed Namespace (splitting
    comma lists, parsing filters and binners); add_cli_args() registers
    the subparser arguments; run_main() forwards the fields to plot().
    """
    output_dir: Path
    fig_dir: Path
    fig_by: list[str]
    row_by: list[str]
    col_by: list[str]
    curve_by: list[str]
    var_x: str
    var_y: str
    filter_by: PlotFilters
    bin_by: PlotBinners
    scale_x: str | None
    scale_y: str | None
    dry_run: bool
    fig_name: str = "FIGURE"
    error_bars: bool = True
    fig_height: float = 6.4
    fig_dpi: int = 300

    parser_name: ClassVar[str] = "plot"
    parser_help: ClassVar[str] = "Plot performance curves from parameter sweep results."

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        """Build a SweepPlotArgs from parsed CLI args. Verifies EXPERIMENT_DIR
        exists, splits --fig-by/--row-by/--col-by/--curve-by comma lists
        (None or empty becomes []), parses --filter-by and --bin-by through
        their parse_str methods, and derives error_bars as the negation of
        --no-error-bars.
        """
        output_dir = Path(args.EXPERIMENT_DIR)
        if not output_dir.exists():
            raise ValueError(f"No parameter sweep results under {output_dir}")

        curve_by = [] if not args.curve_by else args.curve_by.split(",")
        row_by = [] if not args.row_by else args.row_by.split(",")
        col_by = [] if not args.col_by else args.col_by.split(",")
        fig_by = [] if not args.fig_by else args.fig_by.split(",")

        return cls(
            output_dir=output_dir,
            fig_dir=output_dir / args.fig_dir,
            fig_by=fig_by,
            row_by=row_by,
            col_by=col_by,
            curve_by=curve_by,
            var_x=args.var_x,
            var_y=args.var_y,
            filter_by=PlotFilters.parse_str(args.filter_by),
            bin_by=PlotBinners.parse_str(args.bin_by),
            scale_x=args.scale_x,
            scale_y=args.scale_y,
            dry_run=args.dry_run,
            fig_name=args.fig_name,
            error_bars=not args.no_error_bars,
            fig_height=args.fig_height,
            fig_dpi=args.fig_dpi,
        )

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Register the plot subcommand arguments: EXPERIMENT_DIR positional,
        grid selection (--fig-by/--row-by/--col-by/--curve-by), axis variables
        (--var-x default 'total_token_throughput', --var-y default
        'median_ttft_ms'), --filter-by/--bin-by predicate strings,
        --scale-x/--scale-y, --fig-name, --no-error-bars, --fig-height
        (default 6.4), --fig-dpi (default 300), --dry-run. Returns the parser.
        """
        parser.add_argument(
            "EXPERIMENT_DIR",
            type=str,
            help="The directory containing the sweep results to plot.",
        )
        parser.add_argument(
            "--fig-dir",
            type=str,
            default="",
            help="The directory to save the figures, relative to `OUTPUT_DIR`. "
            "By default, the same directory is used.",
        )
        parser.add_argument(
            "--fig-by",
            type=str,
            default="",
            help="A comma-separated list of variables, such that a separate figure "
            "is created for each combination of these variables.",
        )
        parser.add_argument(
            "--row-by",
            type=str,
            default="",
            help="A comma-separated list of variables, such that a separate row "
            "is created for each combination of these variables.",
        )
        parser.add_argument(
            "--col-by",
            type=str,
            default="",
            help="A comma-separated list of variables, such that a separate column "
            "is created for each combination of these variables.",
        )
        parser.add_argument(
            "--curve-by",
            type=str,
            default=None,
            help="A comma-separated list of variables, such that a separate curve "
            "is created for each combination of these variables.",
        )
        parser.add_argument(
            "--var-x",
            type=str,
            default="total_token_throughput",
            help="The variable for the x-axis.",
        )
        parser.add_argument(
            "--var-y",
            type=str,
            default="median_ttft_ms",
            help="The variable for the y-axis",
        )
        parser.add_argument(
            "--filter-by",
            type=str,
            default="",
            help="A comma-separated list of statements indicating values to filter by. "
            "This is useful to remove outliers. "
            "Example: `max_concurrency<1000,max_num_batched_tokens<=4096` means "
            "plot only the points where `max_concurrency` is less than 1000 and "
            "`max_num_batched_tokens` is no greater than 4096.",
        )
        parser.add_argument(
            "--bin-by",
            type=str,
            default="",
            help="A comma-separated list of statements indicating values to bin by. "
            "This is useful to avoid plotting points that are too close together. "
            "Example: `request_throughput%%1` means "
            "use a bin size of 1 for the `request_throughput` variable.",
        )
        parser.add_argument(
            "--scale-x",
            type=str,
            default=None,
            help="The scale to use for the x-axis. "
            "Currently only accepts string values such as 'log' and 'sqrt'. "
            "See also: https://seaborn.pydata.org/generated/seaborn.objects.Plot.scale.html",
        )
        parser.add_argument(
            "--scale-y",
            type=str,
            default=None,
            help="The scale to use for the y-axis. "
            "Currently only accepts string values such as 'log' and 'sqrt'. "
            "See also: https://seaborn.pydata.org/generated/seaborn.objects.Plot.scale.html",
        )
        parser.add_argument(
            "--fig-name",
            type=str,
            default="FIGURE",
            help="Name prefix for the output figure file. "
            "Group data is always appended when present. "
            "Default: 'FIGURE'. Example: --fig-name my_performance_plot",
        )
        parser.add_argument(
            "--no-error-bars",
            action="store_true",
            help="If set, disables error bars on the plot. "
            "By default, error bars are shown.",
        )
        parser.add_argument(
            "--fig-height",
            type=float,
            default=6.4,
            help="Height of each subplot in inches. Default: 6.4",
        )
        parser.add_argument(
            "--fig-dpi",
            type=int,
            default=300,
            help="Resolution of the output figure in dots per inch. Default: 300",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="If set, prints the information about each figure to plot, "
            "then exits without drawing them.",
        )

        return parser


def run_main(args: SweepPlotArgs):
    """Forward every SweepPlotArgs field to plot() and return its result.
    """
    return plot(
        output_dir=args.output_dir,
        fig_dir=args.fig_dir,
        fig_by=args.fig_by,
        row_by=args.row_by,
        col_by=args.col_by,
        curve_by=args.curve_by,
        var_x=args.var_x,
        var_y=args.var_y,
        filter_by=args.filter_by,
        bin_by=args.bin_by,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
        dry_run=args.dry_run,
        fig_name=args.fig_name,
        error_bars=args.error_bars,
        fig_height=args.fig_height,
        fig_dpi=args.fig_dpi,
    )


def main(args: argparse.Namespace):
    """CLI entry point: build a SweepPlotArgs from the parsed Namespace
    and call run_main.
    """
    run_main(SweepPlotArgs.from_cli_args(args))


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=SweepPlotArgs.parser_help)
    SweepPlotArgs.add_cli_args(parser)

    main(parser.parse_args())
