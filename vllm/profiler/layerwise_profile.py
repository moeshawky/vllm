# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Layerwise profiling built on torch.profiler.profile. Records CPU and
CUDA activity with module info, rebuilds a per-module event tree from
the Kineto result, and exposes per-module CUDA time tables.
"""

import copy
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Generic, TypeAlias, TypeVar

from torch._C._autograd import DeviceType, _KinetoEvent, _ProfilerResult
from torch._C._profiler import _EventType, _ExperimentalConfig, _ProfilerEvent
from torch.autograd.profiler import FunctionEvent
from torch.profiler import ProfilerActivity, profile

from vllm.profiler.utils import (
    TablePrinter,
    event_has_module,
    event_is_torch_op,
    event_module_repr,
    event_torch_op_stack_trace,
    indent_string,
)
from vllm.utils.import_utils import PlaceholderModule

try:
    import pandas as pd
except ImportError:
    pd = PlaceholderModule("pandas")


@dataclass
class _ModuleTreeNode:
    """Node in the module event tree built from the Kineto
    event tree. Wraps one _ProfilerEvent and links parent/children
    nodes; trace holds a torch op stack-trace string.
    """
    event: _ProfilerEvent
    parent: "_ModuleTreeNode | None" = None
    children: list["_ModuleTreeNode"] = field(default_factory=list)
    trace: str = ""

    @property
    def is_leaf(self):
        """True when the wrapped event has no children."""
        return self.event.children is None or len(self.event.children) == 0

    @property
    def is_torch_op(self):
        """True when the wrapped event is a TorchOp event."""
        return event_is_torch_op(self.event)

    @property
    def is_cuda(self):
        """True when the event is a Kineto event whose typed
        payload reports device_type CUDA.
        """
        return (
            self.event.tag == _EventType.Kineto
            and self.event.typed[1].device_type == DeviceType.CUDA
        )


@dataclass
class SummaryStatsEntry:
    """Row of the summary stats table: name, cumulative CUDA
    time in microseconds, percentage of total CUDA time, and
    invocation count.
    """
    name: str
    cuda_time_us: float
    pct_cuda_time: float
    invocations: int


@dataclass
class ModelStatsEntry:
    """Row of the model stats table: module/op name, CPU time
    in microseconds, CUDA time in microseconds, percentage of
    total CUDA time, and a torch op stack trace.
    """
    name: str
    cpu_time_us: float
    cuda_time_us: float
    pct_cuda_time: float
    trace: str


StatsEntry: TypeAlias = ModelStatsEntry | SummaryStatsEntry
# Union of the two stats-row dataclasses; parameterizes
# _StatsTreeNode and the flatten/convert helpers.
StatsEntryT = TypeVar("StatsEntryT", bound=StatsEntry)
# TypeVar bound to the StatsEntry union; keeps _StatsTreeNode
# and the tree traversal helpers generic over the row type.


@dataclass
class _StatsTreeNode(Generic[StatsEntryT]):
    """Generic tree node holding one stats-row entry plus
    children/parent links.
    """
    entry: StatsEntryT
    children: list["_StatsTreeNode[StatsEntryT]"] = field(default_factory=list)
    parent: "_StatsTreeNode[StatsEntryT] | None" = None


@dataclass
class LayerwiseProfileResults(profile):
    """Holds profiling results: wraps a torch _ProfilerResult
    (Kineto) and, in __post_init__, builds the correlation map,
    the per-module event tree, and both stats trees. Exposes table
    printing, CSV export, and dict conversion.
    """
    _kineto_results: _ProfilerResult
    _kineto_event_correlation_map: dict[int, list[_KinetoEvent]] = field(init=False)
    _event_correlation_map: dict[int, list[FunctionEvent]] = field(init=False)
    _module_tree: list[_ModuleTreeNode] = field(init=False)
    _model_stats_tree: list[_StatsTreeNode[ModelStatsEntry]] = field(init=False)
    _summary_stats_tree: list[_StatsTreeNode[SummaryStatsEntry]] = field(init=False)

    # profile metadata
    num_running_seqs: int | None = None

    def __post_init__(self):
        """Build the derived structures: correlation map,
        module tree, and both stats trees.
        """
        self._build_correlation_map()
        self._build_module_tree()
        self._build_stats_trees()

    def print_model_table(self, column_widths: dict[str, int] | None = None):
        """Print the model stats tree as a table, dropping rows
        with zero CUDA and zero CPU time.
        """
        _column_widths = dict(
            name=60, cpu_time_us=12, cuda_time_us=12, pct_cuda_time=12, trace=60
        )
        if column_widths:
            _column_widths.update(**column_widths)
        filtered_model_table = [
            (depth, row)
            for depth, row in self._flatten_stats_tree(self._model_stats_tree)
            if row.cuda_time_us > 0 or row.cpu_time_us > 0
        ]
        TablePrinter(ModelStatsEntry, _column_widths).print_table(
            self._indent_row_names_based_on_depth(
                filtered_model_table,
                indent_style=lambda indent: "|" + "-" * indent + " ",
            )
        )

    def print_summary_table(self, column_widths: dict[str, int] | None = None):
        """Print the summary stats tree as a table, dropping
        rows with zero CUDA time.
        """
        _column_widths = dict(
            name=80, cuda_time_us=12, pct_cuda_time=12, invocations=15
        )
        if column_widths:
            _column_widths.update(**column_widths)
        filtered_summary_table = [
            (depth, row)
            for depth, row in self._flatten_stats_tree(self._summary_stats_tree)
            if row.cuda_time_us > 0
        ]
        TablePrinter(SummaryStatsEntry, _column_widths).print_table(
            self._indent_row_names_based_on_depth(
                filtered_summary_table,
                indent_style=lambda indent: "|" + "-" * indent + " ",
            )
        )

    def export_model_stats_table_csv(self, filename: str):
        """Write the flattened model stats tree to a CSV file
        via a pandas DataFrame (asdict per row).
        """
        df = pd.DataFrame(
            [asdict(row) for _, row in self._flatten_stats_tree(self._model_stats_tree)]
        )
        df.to_csv(filename)

    def export_summary_stats_table_csv(self, filename: str):
        """Write the flattened summary stats tree to a CSV file
        via a pandas DataFrame (asdict per row).
        """
        df = pd.DataFrame(
            [
                asdict(row)
                for _, row in self._flatten_stats_tree(self._summary_stats_tree)
            ]
        )
        df.to_csv(filename)

    def convert_stats_to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict with num_running_seqs
        metadata plus the summary and model stats trees.
        """
        return {
            "metadata": {"num_running_seqs": self.num_running_seqs},
            "summary_stats": self._convert_stats_tree_to_dict(self._summary_stats_tree),
            "model_stats": self._convert_stats_tree_to_dict(self._model_stats_tree),
        }

    @staticmethod
    def _indent_row_names_based_on_depth(
        depths_rows: list[tuple[int, StatsEntryT]],
        indent_style: Callable[[int], str] | str = " ",
    ):
        """Static helper: deep-copies each row, prefixes its name
        with an indent derived from depth, and drops rows with
        cuda_time_us == 0.
        """
        indented_rows: list[StatsEntryT] = []
        for depth, row in depths_rows:
            if row.cuda_time_us == 0:
                continue
            indented_row: StatsEntryT = copy.deepcopy(row)
            indented_row.name = indent_string(indented_row.name, depth, indent_style)
            indented_rows.append(indented_row)
        return indented_rows

    def _build_correlation_map(self):
        """Index every Kineto event by correlation_id into
        a defaultdict(list) for later lookup.
        """
        self._kineto_event_correlation_map = defaultdict(list)
        for event in self._kineto_results.events():
            self._kineto_event_correlation_map[event.correlation_id()].append(event)

    def _build_module_tree(self):
        """DFS over the Kineto experimental event tree to
        build _module_tree: a node for each event carrying module
        info, plus leaf nodes for torch ops under the nearest module
        ancestor. Only events with start_tid == 1 are considered.
        """
        self._module_tree = []
        event_tree = self._kineto_results.experimental_event_tree()

        def _df_traversal(
            event: _ProfilerEvent, curr_node: _ModuleTreeNode | None = None
        ):
            # For the tensor parallel case for now only look at task 1
            if event.start_tid != 1:
                return

            if event_has_module(event):
                node = _ModuleTreeNode(event=event, parent=curr_node)
                if curr_node:
                    curr_node.children.append(node)
                else:
                    self._module_tree.append(node)
                curr_node = node

            is_leaf = event.children is None or len(event.children) == 0
            if is_leaf and curr_node:
                node = _ModuleTreeNode(
                    event=event,
                    parent=curr_node,
                    trace=event_torch_op_stack_trace(
                        event, until=lambda x: event_has_module(x)
                    ),
                )
                curr_node.children.append(node)
                curr_node = node

            for child in event.children:
                _df_traversal(child, curr_node)

        for root in event_tree:
            _df_traversal(root)

    def _get_kineto_gpu_event(self, node: _ModuleTreeNode):
        """Look up the CUDA Kineto event correlated with a
        module tree node.
        """
        if node.event.tag != _EventType.Kineto:
            return None
        correlated_kineto_events = self._kineto_event_correlation_map.get(
            node.event.correlation_id, []
        )
        iterator = (
            x
            for x in correlated_kineto_events
            if x.device_type() == DeviceType.CUDA and x.name() == node.event.name
        )
        return next(iterator, None)

    def _cumulative_cuda_time(self, node: _ModuleTreeNode):
        "Return cuda time in microseconds"

        def _cumulative_cuda_time_recursive(node: _ModuleTreeNode):
            if node.is_leaf and (gpu_kineto_event := self._get_kineto_gpu_event(node)):
                return gpu_kineto_event.duration_ns() / 1000.0
            else:
                cumulative_cuda_time = 0
                for child in node.children:
                    cumulative_cuda_time += _cumulative_cuda_time_recursive(child)
                return cumulative_cuda_time

        return _cumulative_cuda_time_recursive(node)

    def _total_cuda_time(self):
        """Sum of _cumulative_cuda_time over all module tree
        roots; the denominator used for pct_cuda_time.
        """
        return sum([self._cumulative_cuda_time(root) for root in self._module_tree])

    def _build_stats_trees(self):
        """Populate _summary_stats_tree and _model_stats_tree
        from _module_tree. Summary rows aggregate repeated module
        paths; model rows are one node per module/op occurrence.
        """
        summary_dict: dict[tuple[str, ...], _StatsTreeNode[SummaryStatsEntry]] = {}
        total_cuda_time = self._total_cuda_time()

        def pct_cuda_time(cuda_time_us):
            return (cuda_time_us / total_cuda_time) * 100

        def build_summary_stats_tree_df(
            node: _ModuleTreeNode,
            parent: _StatsTreeNode[SummaryStatsEntry] | None = None,
            summary_trace: tuple[str, ...] = (),
        ) -> _StatsTreeNode[SummaryStatsEntry] | None:
            if event_has_module(node.event):
                name = event_module_repr(node.event)
                cuda_time_us = self._cumulative_cuda_time(node)
            elif gpu_kineto_event := self._get_kineto_gpu_event(node):
                name = gpu_kineto_event.name()
                cuda_time_us = gpu_kineto_event.duration_ns() / 1000.0
            else:
                return None

            summary_trace = summary_trace + (name,)
            if summary_trace in summary_dict:
                entry = summary_dict[summary_trace].entry
                entry.cuda_time_us += cuda_time_us
                entry.invocations += 1
                entry.pct_cuda_time = pct_cuda_time(entry.cuda_time_us)
            else:
                new_node = _StatsTreeNode(
                    entry=SummaryStatsEntry(
                        name=name,
                        cuda_time_us=cuda_time_us,
                        pct_cuda_time=pct_cuda_time(cuda_time_us),
                        invocations=1,
                    ),
                    parent=parent,
                )
                if parent:
                    parent.children.append(new_node)
                summary_dict[summary_trace] = new_node

            for child in node.children:
                build_summary_stats_tree_df(
                    child, summary_dict[summary_trace], summary_trace
                )

            return summary_dict[summary_trace]

        self._summary_stats_tree = []
        for root in self._module_tree:
            summary_node = build_summary_stats_tree_df(root)
            if summary_node is not None:
                self._summary_stats_tree.append(summary_node)

        def build_model_stats_tree_df(
            node: _ModuleTreeNode,
            parent: _StatsTreeNode[ModelStatsEntry] | None = None,
        ) -> _StatsTreeNode[ModelStatsEntry] | None:
            if event_has_module(
                node.event,
            ):
                name = event_module_repr(node.event)
                cuda_time_us = self._cumulative_cuda_time(node)
                cpu_time_us = node.event.duration_time_ns / 1000
                trace = ""
            elif gpu_kineto_event := self._get_kineto_gpu_event(node):
                name = gpu_kineto_event.name()
                cuda_time_us = gpu_kineto_event.duration_ns() / 1000.0
                cpu_time_us = 0
                trace = node.trace
            else:
                return None

            new_node = _StatsTreeNode(
                entry=ModelStatsEntry(
                    name=name,
                    cpu_time_us=cpu_time_us,
                    cuda_time_us=cuda_time_us,
                    pct_cuda_time=pct_cuda_time(cuda_time_us),
                    trace=trace,
                ),
                parent=parent,
            )
            if parent:
                parent.children.append(new_node)

            for child in node.children:
                build_model_stats_tree_df(child, new_node)

            return new_node

        self._model_stats_tree = []
        for root in self._module_tree:
            model_node = build_model_stats_tree_df(root)
            if model_node is not None:
                self._model_stats_tree.append(model_node)

    def _flatten_stats_tree(
        self, tree: list[_StatsTreeNode[StatsEntryT]]
    ) -> list[tuple[int, StatsEntryT]]:
        """DFS-flatten a stats tree into a list of (depth, entry)
        pairs for table printing.
        """
        entries: list[tuple[int, StatsEntryT]] = []

        def df_traversal(node: _StatsTreeNode[StatsEntryT], depth: int = 0):
            entries.append((depth, node.entry))
            for child in node.children:
                df_traversal(child, depth=depth + 1)

        for root in tree:
            df_traversal(root)

        return entries

    def _convert_stats_tree_to_dict(
        self, tree: list[_StatsTreeNode[StatsEntryT]]
    ) -> list[dict[str, Any]]:
        """DFS-convert a stats tree into a nested list of
        {'entry': asdict, 'children': [...]} dicts.
        """
        root_dicts: list[dict[str, Any]] = []

        def df_traversal(
            node: _StatsTreeNode[StatsEntryT], curr_json_list: list[dict[str, Any]]
        ):
            curr_json_list.append({"entry": asdict(node.entry), "children": []})
            for child in node.children:
                df_traversal(child, curr_json_list[-1]["children"])

        for root in tree:
            df_traversal(root, root_dicts)

        return root_dicts


class layerwise_profile(profile):
    """Profiler context manager subclassing torch.profiler.profile.
    Configures CPU+CUDA activities, record_shapes, with_stack,
    with_modules, and a verbose _ExperimentalConfig. On __exit__
    it builds a LayerwiseProfileResults from self.profiler.kineto_results.
    """
    def __init__(self, num_running_seqs: int | None = None):
        """
        layerwise profile constructor.

        Args:
            num_running_seqs (Optional[int], optional): When given,
                num_running_seqs will be passed to LayerProfileResults
                for metadata update. Defaults to None.
        """
        super().__init__(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            with_modules=True,
            experimental_config=_ExperimentalConfig(verbose=True),
        )

        self.num_running_seqs = num_running_seqs

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the torch profiler via super().__exit__, then
        construct LayerwiseProfileResults from self.profiler.kineto_results.
        """
        super().__exit__(exc_type, exc_val, exc_tb)
        self.results = LayerwiseProfileResults(
            self.profiler.kineto_results, num_running_seqs=self.num_running_seqs
        )
