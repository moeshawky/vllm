# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for the benchmark sweep CLI.

Sanitizes benchmark run labels into file-system-safe names so
sweep outputs can be written to disk (see sanitize_filename).
"""


def sanitize_filename(filename: str) -> str:
    """Map a benchmark run label into a file-system-safe name.

    Replaces "/" with "_", then replaces ".." with "__", then strips
    single and double quote characters from the two ends only. Quote
    characters inside the label are left in place.

    Args:
        filename: Raw run label (may contain "/", "..", and quotes).

    Returns:
        The mapped label, safe to use as a file name on POSIX hosts.
    """
    return filename.replace("/", "_").replace("..", "__").strip("'").strip('"')
