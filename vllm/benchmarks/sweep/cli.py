# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""argparse wiring for the `vllm bench sweep` subcommand family. add_cli_args
  adds one required subparser per sweep command (serve, serve-workload, startup,
  plot, plot-pareto) and pins the command's main function onto the parsed namespace;
  main invokes that pinned function. Wired up by BenchmarkSweepSubcommand in
  vllm/entrypoints/cli/benchmark/sweep.py."""
import argparse

from vllm.entrypoints.serve.utils.api_utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.utils.argparse_utils import FlexibleArgumentParser

from .plot import SweepPlotArgs
from .plot import main as plot_main
from .plot_pareto import SweepPlotParetoArgs
from .plot_pareto import main as plot_pareto_main
from .serve import SweepServeArgs
from .serve import main as serve_main
from .serve_workload import SweepServeWorkloadArgs
from .serve_workload import main as serve_workload_main
from .startup import SweepStartupArgs
from .startup import main as startup_main

# One (args_class, main_function) pair per registered sweep subcommand;
# iteration order defines the help listing order in `vllm bench sweep --help`.
SUBCOMMANDS = (
    (SweepServeArgs, serve_main),
    (SweepServeWorkloadArgs, serve_workload_main),
    (SweepStartupArgs, startup_main),
    (SweepPlotArgs, plot_main),
    (SweepPlotParetoArgs, plot_pareto_main),
)


def add_cli_args(parser: FlexibleArgumentParser):
    """Adds a required subparser group to the `vllm bench` parser, one subcommand
      per entry in SUBCOMMANDS. Each subparser gets the command's parser_name,
      parser_help, a formatted usage string, the command's own arguments via
      cmd.add_cli_args, the shared VLLM_SUBCMD_PARSER_EPILOG, and a dispatch_function
      default pointing at the command's main."""
    subparsers = parser.add_subparsers(required=True, dest="sweep_type")

    for cmd, entrypoint in SUBCOMMANDS:
        cmd_subparser = subparsers.add_parser(
            cmd.parser_name,
            description=cmd.parser_help,
            usage=f"vllm bench sweep {cmd.parser_name} [options]",
        )
        cmd_subparser.set_defaults(dispatch_function=entrypoint)
        cmd.add_cli_args(cmd_subparser)
        cmd_subparser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(
            subcmd=f"sweep {cmd.parser_name}"
        )


def main(args: argparse.Namespace):
    """Dispatches to the subcommand's main: looks up dispatch_function
      on the parsed namespace (set by add_cli_args via set_defaults) and calls it
      with the namespace."""
    args.dispatch_function(args)
