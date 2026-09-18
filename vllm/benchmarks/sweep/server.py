# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Context-manager wrapper around one external inference server process
for the sweep. start() spawns the server command in its own session with
VLLM_SERVER_DEV_MODE=1; wait_until_ready polls GET /health until 200 or
timeout; run_subcommand runs the benchmark against the live server;
stop() SIGKILLs the whole process group; reset_caches POSTs the dev-mode
cache-reset endpoints between runs.
"""
import contextlib
import os
import signal
import subprocess
import time
from types import TracebackType

import requests
from typing_extensions import Self


class ServerProcess:
    """Wrapper binding the server lifecycle to a Python context:
    __enter__ spawns the process, __exit__ kills the process group even
    on exception. Owns the health poll, the one-shot subcommand runner,
    the post-bench cache reset, and reconstructs the server base URL
    by scanning its own command line.
    """
    # Dev-mode-only endpoints (the server must run with
    # VLLM_SERVER_DEV_MODE=1, which start() sets) that reset
    # the prefix cache, multi-modal cache, and encoder cache
    # between benchmark runs.
    VLLM_RESET_CACHE_ENDPOINTS = [
        "/reset_prefix_cache",
        "/reset_mm_cache",
        "/reset_encoder_cache",
    ]

    def __init__(
        self,
        server_cmd: list[str],
        after_bench_cmd: list[str],
        *,
        show_stdout: bool,
    ) -> None:
        """Stores the server command, the optional post-bench command,
        and the stdout flag. The process handle is created later in start().
        """
        super().__init__()

        self.server_cmd = server_cmd
        self.after_bench_cmd = after_bench_cmd
        self.show_stdout = show_stdout

    def __enter__(self) -> Self:
        """Spawns the server and returns the wrapper, so `with ServerProcess(...)
        as server:` covers spawn through group-kill.
        """
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        """Stops the server on context exit, including exception paths,
        so a failing benchmark still reclaims the process group.
        """
        self.stop()

    def start(self):
        """Spawns server_cmd with start_new_session=True so the child
        leads its own session/process group (which stop() later kills
        with os.killpg), merges VLLM_SERVER_DEV_MODE=1 into the
        environment for the dev-only cache-reset endpoints, and hides
        stdout unless show_stdout.
        """
        # Create new process for clean termination
        self._server_process = subprocess.Popen(
            self.server_cmd,
            start_new_session=True,
            stdout=None if self.show_stdout else subprocess.DEVNULL,
            # Need `VLLM_SERVER_DEV_MODE=1` for `_reset_caches`
            env=os.environ | {"VLLM_SERVER_DEV_MODE": "1"},
        )

    def stop(self):
        """If the spawned process is still alive, SIGKILLs its entire
        process group (API server plus engine children) via os.killpg on
        the leader pgid; ProcessLookupError is suppressed for the race
        where the group already vanished.
        """
        server_process = self._server_process

        if server_process.poll() is None:
            # In case only some processes have been terminated
            with contextlib.suppress(ProcessLookupError):
                # We need to kill both API Server and Engine processes
                os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)

    def run_subcommand(self, cmd: list[str]):
        """Runs a one-shot command (the benchmark) with the same stdout
        policy; check=True raises CalledProcessError on a non-zero exit.
        """
        return subprocess.run(
            cmd,
            stdout=None if self.show_stdout else subprocess.DEVNULL,
            check=True,
        )

    def after_bench(self) -> None:
        """Runs the user-supplied after_bench_cmd when present; otherwise
        falls back to reset_caches().
        """
        if not self.after_bench_cmd:
            self.reset_caches()
            return

        self.run_subcommand(self.after_bench_cmd)

    def _get_vllm_server_address(self) -> str:
        """Reconstructs the server base URL by scanning server_cmd for a
        space-separated --host flag and a -p/--port flag; falls back
        to localhost:8000 (vllm serve defaults) when either is absent.
        """
        server_cmd = self.server_cmd

        for host_key in ("--host",):
            if host_key in server_cmd:
                host = server_cmd[server_cmd.index(host_key) + 1]
                break
        else:
            host = "localhost"

        for port_key in ("-p", "--port"):
            if port_key in server_cmd:
                port = int(server_cmd[server_cmd.index(port_key) + 1])
                break
        else:
            port = 8000  # The default value in vllm serve

        return f"http://{host}:{port}"

    def is_server_ready(self) -> bool:
        """GETs {base}/health; ready means HTTP 200. Any requests.RequestException
        (refused connection, DNS failure, timeout) counts as not ready.
        """
        server_address = self._get_vllm_server_address()
        try:
            response = requests.get(f"{server_address}/health")
            return response.status_code == 200
        except requests.RequestException:
            return False

    def wait_until_ready(self, timeout: int) -> None:
        """Polls is_server_ready once per second on a monotonic clock. Raises
        RuntimeError (with the return code) if the process died mid-wait,
        and TimeoutError once the budget is spent.
        """
        start_time = time.monotonic()
        while not self.is_server_ready():
            # Check if server process has crashed
            if self._server_process.poll() is not None:
                returncode = self._server_process.returncode
                raise RuntimeError(
                    f"Server process crashed with return code {returncode}"
                )
            if time.monotonic() - start_time > timeout:
                raise TimeoutError(
                    f"Server failed to become ready within {timeout} seconds."
                )
            time.sleep(1)

    def reset_caches(self) -> None:
        """Dispatches on the server binary name (server_cmd[0], matched with
        endswith so /bin/... paths work): vllm POSTs each reset endpoint
        and raises on non-2xx; infinity_emb raises NotImplementedError
        when --vector-disk-cache is configured (no reset API);
        anything else raises with a pointer to --after-bench-cmd.
        """
        server_cmd = self.server_cmd

        # Use `.endswith()` to match `/bin/...`
        if server_cmd[0].endswith("vllm"):
            server_address = self._get_vllm_server_address()
            print(f"Resetting caches at {server_address}")

            for endpoint in self.VLLM_RESET_CACHE_ENDPOINTS:
                res = requests.post(server_address + endpoint)
                res.raise_for_status()
        elif server_cmd[0].endswith("infinity_emb"):
            if "--vector-disk-cache" in server_cmd:
                raise NotImplementedError(
                    "Infinity server uses caching but does not expose a method "
                    "to reset the cache"
                )
        else:
            raise NotImplementedError(
                f"No implementation of `reset_caches` for `{server_cmd[0]}` server. "
                "Please specify a custom command via `--after-bench-cmd`."
            )
