import ast
import json
from pathlib import Path

import numpy as np


NOTEBOOK = Path(__file__).with_name(
    "09_counterfactual_value_equivalent_adaln.ipynb"
)


def main():
    payload = json.loads(NOTEBOOK.read_text())
    code_cells = [
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    ]
    for index, source in enumerate(code_cells):
        try:
            ast.parse(source)
        except SyntaxError as error:
            raise AssertionError(
                f"code cell {index} has invalid Python: {error}"
            ) from error

    joined = "\n".join(code_cells)
    required = [
        'REPO_COMMIT = "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"',
        "matched_cave_action_path",
        "shuffled_cave_action_path",
        "latent_only_action_path",
        "action_encoder",
        "adaLN_modulation[1]",
        "differentiable_unroll",
        "effect",
        "deterministic_outcome_permutation",
        "EVALUATION_PROJECTION_SEED",
        'RESULT_ZIP: {result_zip}',
        'RUN_STATUS:',
        'colab_files.download',
    ]
    missing = [needle for needle in required if needle not in joined]
    if missing:
        raise AssertionError(f"missing required Stage 9 elements: {missing}")
    if 'OUTPUT_DIR = "/content/counterfactual_faithfulness_stage9"' not in joined:
        raise AssertionError("Stage 9 output directory is not fixed")
    if 'Path("/content/stage9_result_bundle.zip")' not in joined:
        raise AssertionError("Stage 9 bundle path is not fixed")
    if (
        "EVALUATION_PROJECTION_SEED"
        not in joined.split("# Phase D")[1]
    ):
        raise AssertionError("fresh evaluation projection is not used")
    if len(payload["cells"]) != 11:
        raise AssertionError(f"expected 11 cells, found {len(payload['cells'])}")

    # The same-state effect target must be invariant to a shared endpoint bias.
    rng = np.random.default_rng(9401)
    endpoint = rng.normal(size=(10, 6, 4))
    common_bias = rng.normal(size=(1, 6, 4))
    original_effect = endpoint - endpoint[:1]
    biased_effect = (endpoint + common_bias) - (endpoint + common_bias)[:1]
    np.testing.assert_allclose(original_effect, biased_effect)

    # A fixed null branch plus shuffled non-null branches must preserve the
    # outcome multiset while destroying action correspondence.
    non_null = np.arange(1, 10)
    rng.shuffle(non_null)
    permutation = np.concatenate([[0], non_null])
    assert permutation[0] == 0
    assert sorted(permutation.tolist()) == list(range(10))

    # Verify the decision-gap lemma used to motivate the intervention.
    true_cost = np.asarray([0.10, 0.35, 0.70])
    value_error_bound = 0.08
    predicted_cost = true_cost + np.asarray(
        [value_error_bound, -value_error_bound, value_error_bound]
    )
    true_gap = np.partition(true_cost, 1)[1] - true_cost.min()
    assert true_gap > 2 * value_error_bound
    assert int(np.argmin(predicted_cost)) == int(np.argmin(true_cost))
    print(
        json.dumps(
            {
                "notebook": str(NOTEBOOK),
                "cells": len(payload["cells"]),
                "code_cells": len(code_cells),
                "status": "ok",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
