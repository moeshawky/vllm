# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Data types for `vllm bench sweep` parameter sweep inputs: ParameterSweep
  loads a sweep from JSON (dict or list form) and ParameterSweepItem wraps
  one record of parameter key/value pairs, providing JSON/CLI key-spelling
  lookup and application of the parameters onto a base command line."""
import json
import os


class ParameterSweep(list["ParameterSweepItem"]):
    """List of ParameterSweepItem records describing one parameter sweep.
      Instances are built from JSON via read_json, which accepts either a dict mapping
      names to parameter dicts or a plain list of parameter dicts."""
    @classmethod
    def read_json(cls, filepath: os.PathLike):
        """Loads and parses a JSON sweep file. A top-level dict is treated as
          name-to-parameters (read_from_dict); a top-level list is treated as parameter
          records (from_records)."""
        with open(filepath, "rb") as f:
            data = json.load(f)

        # Support both list and dict formats
        if isinstance(data, dict):
            return cls.read_from_dict(data)

        return cls.from_records(data)

    @classmethod
    def read_from_dict(cls, data: dict[str, dict[str, object]]):
        """
        Read parameter sweep from a dict format where keys are names.

        Example:
            {
                "experiment1": {"max_tokens": 100, "temperature": 0.7},
                "experiment2": {"max_tokens": 200, "temperature": 0.9}
            }
        """
        records = [{"_benchmark_name": name, **params} for name, params in data.items()]
        return cls.from_records(records)

    @classmethod
    def from_records(cls, records: list[dict[str, object]]):
        """Builds a ParameterSweep from a list of dicts. Raises TypeError if
          the input is not a list, and ValueError if any _benchmark_name values
          are duplicated. Each record is wrapped via ParameterSweepItem.from_record."""
        if not isinstance(records, list):
            raise TypeError(
                f"The parameter sweep should be a list of dictionaries, "
                f"but found type: {type(records)}"
            )

        # Validate that all _benchmark_name values are unique if provided
        names = [r["_benchmark_name"] for r in records if "_benchmark_name" in r]
        if names and len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(
                f"Duplicate _benchmark_name values found: {set(duplicates)}. "
                f"All _benchmark_name values must be unique."
            )

        return cls(ParameterSweepItem.from_record(record) for record in records)


class ParameterSweepItem(dict[str, object]):
    """Dict mapping one sweep configuration's parameter names to values,
      plus the optional _benchmark_name key. Provides key lookup that tolerates JSON/CLI
      spelling differences (underscore vs dash) and application of the parameters
      onto a base command list."""
    @classmethod
    def from_record(cls, record: dict[str, object]):
        """Wraps a plain dict as a ParameterSweepItem. Raises TypeError if the
          record is not a dict."""
        if not isinstance(record, dict):
            raise TypeError(
                f"Each item in the parameter sweep should be a dictionary, "
                f"but found type: {type(record)}"
            )

        return cls(record)

    def __or__(self, other: dict[str, object], /) -> "ParameterSweepItem":  # type: ignore[override]
        """Merges another dict into this item and returns the result as a new
          ParameterSweepItem, so `item | other` stays typed as ParameterSweepItem
          instead of plain dict."""
        return ParameterSweepItem(super().__or__(other))

    @property
    def name(self) -> str:
        """
        Get the name for this parameter sweep item.

        Returns the '_benchmark_name' field if present, otherwise returns a text
        representation of all parameters.
        """
        if "_benchmark_name" in self:
            return str(self["_benchmark_name"])

        return self.as_text(sep="-")

    # In JSON, we prefer "_"
    def _iter_param_key_candidates(self, param_key: str):
        # Inner config arguments are not converted by the CLI
        if "." in param_key:
            prefix, rest = param_key.split(".", 1)
            for prefix_candidate in self._iter_param_key_candidates(prefix):
                yield prefix_candidate + "." + rest

            return

        yield param_key
        yield param_key.replace("-", "_")
        yield param_key.replace("_", "-")

    # In CLI, we prefer "-"
    def _iter_cmd_key_candidates(self, param_key: str):
        for k in reversed(tuple(self._iter_param_key_candidates(param_key))):
            yield "--" + k

    def _normalize_cmd_key(self, param_key: str):
        """Resolves to the first candidate of the reversed iteration, i.e. the
          dash-separated CLI spelling; for a key with no dashes or underscores all three
          candidates are identical."""
        return next(self._iter_cmd_key_candidates(param_key))

    def has_param(self, param_key: str) -> bool:
        """Returns True if any spelling candidate of param_key (dash/underscore
          variants, recursing on dotted prefixes) is present as a key in the item."""
        return any(k in self for k in self._iter_param_key_candidates(param_key))

    def _normalize_cmd_kv_pair(self, k: str, v: object) -> list[str]:
        """
        Normalize a key-value pair into command-line arguments.

        Returns a list containing either:
        - A single element for boolean flags (e.g., ['--flag'] or ['--flag=true'])
        - Two elements for key-value pairs (e.g., ['--key', 'value'])
        """
        if isinstance(v, bool):
            # For nested params (containing "."), use =true/false syntax
            if "." in k:
                return [f"{self._normalize_cmd_key(k)}={'true' if v else 'false'}"]
            else:
                return [self._normalize_cmd_key(k if v else "no-" + k)]
        else:
            return [self._normalize_cmd_key(k), str(v)]

    def apply_to_cmd(self, cmd: list[str]) -> list[str]:
        """Returns a copy of the base command list with this item's parameters
          applied. For each key (skipping _benchmark_name), if a matching flag token
          exists the in-place value is replaced; otherwise the key/value pair is
          appended. Dict values are JSON-serialized; boolean values become flags
          (with no- prefix for False)."""
        cmd = list(cmd)

        for k, v in self.items():
            # Skip the '_benchmark_name' field, not a parameter
            if k == "_benchmark_name":
                continue

            # Serialize dict values as JSON
            if isinstance(v, dict):
                v = json.dumps(v)

            for k_candidate in self._iter_cmd_key_candidates(k):
                try:
                    k_idx = cmd.index(k_candidate)

                    # Replace existing parameter
                    normalized = self._normalize_cmd_kv_pair(k, v)
                    if len(normalized) == 1:
                        # Boolean flag
                        cmd[k_idx] = normalized[0]
                    else:
                        # Key-value pair
                        cmd[k_idx] = normalized[0]
                        cmd[k_idx + 1] = normalized[1]

                    break
                except ValueError:
                    continue
            else:
                # Add new parameter
                cmd.extend(self._normalize_cmd_kv_pair(k, v))

        return cmd

    def as_text(self, sep: str = ", ") -> str:
        """Renders all parameters except _benchmark_name as 'key=value' pairs
          joined by sep (default ', '); used by the name property as the fallback
          display string."""
        return sep.join(f"{k}={v}" for k, v in self.items() if k != "_benchmark_name")
