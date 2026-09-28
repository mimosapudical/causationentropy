"""Audit only_return_significant semantics for sparse-selection methods.

The audit forces Information-LASSO to select one candidate and forces the final
shuffle test for that edge to fail.  If only_return_significant=True still
returns the edge, the public option is not acting as a final significance
filter for Information-LASSO.

This is diagnostic only.
"""

import contextlib
import io
import json
from unittest.mock import patch

import numpy as np

from causationentropy.core.discovery import discover_network


def run_audit():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(40, 2))

    fake_shuffle = {
        "Threshold": 1.0,
        "Value": 0.1,
        "Pass": False,
        "P_value": 0.9,
    }

    with patch(
        "causationentropy.core.discovery."
        "information_lasso_optimal_causation_entropy",
        return_value=[0],
    ), patch(
        "causationentropy.core.discovery.conditional_mutual_information",
        return_value=0.1,
    ), patch(
        "causationentropy.core.discovery.shuffle_test",
        return_value=fake_shuffle,
    ):
        # discover_network reports target progress to stdout. Keep this audit's
        # stdout machine-readable so run_path_b_v2 can persist valid JSON.
        with contextlib.redirect_stdout(io.StringIO()):
            graph_significant_only = discover_network(
                data,
                method="information_lasso",
                max_lag=1,
                n_shuffles=5,
                only_return_significant=True,
            )

    return {
        "forced_final_shuffle_pass": False,
        "only_return_significant": True,
        "returned_edge_count": graph_significant_only.number_of_edges(),
        "returned_edges": [
            {
                "source": str(source),
                "target": str(target),
                **attrs,
            }
            for source, target, attrs in graph_significant_only.edges(data=True)
        ],
        "implementation_changed": False,
    }


def main():
    print(json.dumps(run_audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
