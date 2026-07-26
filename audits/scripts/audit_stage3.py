import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


BUNDLE = Path(
    os.environ.get("STAGE3_BUNDLE", "stage3_result_bundle")
).expanduser()
TIE = 1e-9
SEED = 71
BOOTSTRAP_REPS = 2000
READOUTS = [
    "latent_distance",
    "linear_pose",
    "action_blind",
    "linear_pose_shuffled",
    "oracle_pose",
]


def load_json(name):
    with (BUNDLE / name).open() as handle:
        return json.load(handle)


def bootstrap_mean(values, groups, repetitions, seed):
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    finite = np.isfinite(values)
    values = values[finite]
    groups = groups[finite]
    unique = np.unique(groups)
    if len(unique) == 0:
        return {
            "estimate": np.nan,
            "low": np.nan,
            "high": np.nan,
            "n_clusters": 0,
            "n_bootstrap": repetitions,
        }
    grouped = [values[groups == group] for group in unique]
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    for index in range(repetitions):
        sampled = rng.integers(0, len(unique), size=len(unique))
        draws[index] = np.mean(
            np.concatenate([grouped[item] for item in sampled])
        )
    return {
        "estimate": float(np.mean(values)),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "n_clusters": int(len(unique)),
        "n_bootstrap": int(repetitions),
    }


def unit_metrics(frame):
    frame = frame.sort_values("action")
    truth = frame["true_cost"].to_numpy(dtype=float)
    prediction = frame["predicted_cost"].to_numpy(dtype=float)
    selected = int(np.argmin(prediction))
    oracle = int(np.argmin(truth))
    best = float(np.min(truth))
    chosen = float(truth[selected])
    spread = float(np.max(truth) - best)
    regret = chosen - best
    normalized_regret = regret / spread if spread > TIE else 0.0
    left, right = np.triu_indices(len(truth), k=1)
    true_margin = truth[left] - truth[right]
    predicted_margin = prediction[left] - prediction[right]
    valid = np.abs(true_margin) > TIE
    credit = np.full(len(left), np.nan)
    same = np.sign(true_margin) == np.sign(predicted_margin)
    credit[valid & same] = 1.0
    credit[valid & (np.abs(predicted_margin) <= TIE)] = 0.5
    credit[valid & np.isnan(credit)] = 0.0
    weights = np.abs(true_margin)
    pairwise = float(np.nanmean(credit)) if np.any(valid) else np.nan
    weighted = (
        float(np.nansum(weights * credit) / np.sum(weights[valid]))
        if np.any(valid)
        else np.nan
    )
    margin_scale = (
        float(np.sqrt(np.mean(true_margin[valid] ** 2)))
        if np.any(valid)
        else 0.0
    )
    margin_mse = (
        float(np.mean((predicted_margin[valid] - true_margin[valid]) ** 2))
        if np.any(valid)
        else np.nan
    )
    notebook_normalized_margin_rmse = (
        float(np.sqrt(margin_mse) / margin_scale)
        if margin_scale > TIE
        else np.nan
    )
    pose_error = np.nan
    physical_cost_rmse = np.nan
    if frame["readout"].iloc[0] == "linear_pose":
        environment = frame["environment"].iloc[0]
        dimensions = 4 if environment == "PushT" else 2
        truth_pose = frame[
            [f"true_pose_{index}" for index in range(dimensions)]
        ].to_numpy(dtype=float)
        predicted_pose = frame[
            [f"predicted_pose_{index}" for index in range(dimensions)]
        ].to_numpy(dtype=float)
        if environment == "PushT":
            angle_prediction = np.arctan2(
                predicted_pose[:, 2], predicted_pose[:, 3]
            )
            angle_truth = np.arctan2(truth_pose[:, 2], truth_pose[:, 3])
            angle_error = np.arctan2(
                np.sin(angle_prediction - angle_truth),
                np.cos(angle_prediction - angle_truth),
            )
            pieces = np.column_stack(
                [
                    predicted_pose[:, :2] - truth_pose[:, :2],
                    angle_error / np.pi,
                ]
            )
        else:
            pieces = predicted_pose[:, :2] - truth_pose[:, :2]
        pose_error = float(np.mean(np.linalg.norm(pieces, axis=-1)))
        physical_cost_rmse = float(
            np.sqrt(np.mean((prediction - truth) ** 2))
        )
    return pd.Series(
        {
            "selected_action": selected,
            "oracle_action": oracle,
            "top1_correct": float(chosen <= best + TIE),
            "normalized_regret": normalized_regret,
            "pairwise_accuracy": pairwise,
            "weighted_pairwise_accuracy": weighted,
            "normalized_margin_rmse": notebook_normalized_margin_rmse,
            "true_cost_spread": spread,
            "pose_error": pose_error,
            "physical_cost_rmse": physical_cost_rmse,
            "selected_interaction_type": frame["interaction_type"].iloc[
                selected
            ],
            "oracle_interaction_type": frame["interaction_type"].iloc[oracle],
        }
    )


def interval_close(left, right, tolerance=1e-12):
    keys = ["estimate", "low", "high"]
    return all(
        (np.isnan(left[key]) and np.isnan(right[key]))
        or abs(left[key] - right[key]) <= tolerance
        for key in keys
    ) and left["n_clusters"] == right["n_clusters"]


action = pd.read_csv(BUNDLE / "action_predictions.csv")
pair = pd.read_csv(BUNDLE / "pair_metrics.csv")
summary = pd.read_csv(BUNDLE / "metrics_summary.csv")
rankings = pd.read_csv(BUNDLE / "model_rankings.csv")
regression_predictions = pd.read_csv(
    BUNDLE / "held_out_regression_predictions.csv"
)
decision = load_json("stage3_decision.json")
candidate_summary = load_json("candidate_design_summary.json")
restore = load_json("restore_test.json")
tasks = pd.DataFrame(load_json("tasks.json"))
split_manifest = load_json("split_manifest.json")
probe_manifest = load_json("probe_manifest.json")

row_counts = {
    "action": len(action),
    "pair": len(pair),
    "summary": len(summary),
    "rankings": len(rankings),
    "regression_predictions": len(regression_predictions),
}
expected_row_counts = {
    "action": 72_000,
    "pair": 324_000,
    "summary": 60,
    "rankings": 12,
    "regression_predictions": 1_440,
}

action_group = [
    "environment",
    "state_id",
    "task_id",
    "model",
    "probe_seed",
    "readout",
    "horizon",
]
action_key = action_group + ["action"]
pair_key = action_group + ["pair_left", "pair_right"]
summary_key = ["environment", "model", "readout", "horizon"]
ranking_key = ["environment", "horizon", "model"]
regression_key = [
    "environment",
    "state_id",
    "task_id",
    "model",
    "probe_seed",
    "horizon",
]

duplicates = {
    "action": int(action.duplicated(action_key).sum()),
    "pair": int(pair.duplicated(pair_key).sum()),
    "summary": int(summary.duplicated(summary_key).sum()),
    "rankings": int(rankings.duplicated(ranking_key).sum()),
    "regression_predictions": int(
        regression_predictions.duplicated(regression_key).sum()
    ),
}
group_sizes = {
    "actions": action.groupby(action_group).size().value_counts().to_dict(),
    "pairs": pair.groupby(action_group).size().value_counts().to_dict(),
}

task_split_checks = {}
for environment in ["PushT", "Wall"]:
    env_tasks = tasks[tasks["environment"] == environment]
    manifest = split_manifest["environments"][environment]
    task_split_checks[environment] = {
        "task_counts": env_tasks["split"].value_counts().sort_index().to_dict(),
        "manifest_state_counts": {
            name: len(payload["state_ids"])
            for name, payload in manifest.items()
        },
        "manifest_task_counts": {
            name: len(payload["task_ids"])
            for name, payload in manifest.items()
        },
        "task_sets_disjoint": (
            sum(
                len(payload["task_ids"]) for payload in manifest.values()
            )
            == len(
                set().union(
                    *(set(payload["task_ids"]) for payload in manifest.values())
                )
            )
        ),
        "state_sets_disjoint": (
            sum(
                len(payload["state_ids"]) for payload in manifest.values()
            )
            == len(
                set().union(
                    *(
                        set(payload["state_ids"])
                        for payload in manifest.values()
                    )
                )
            )
        ),
        "exported_tasks_are_final_only": set(
            action.loc[action["environment"] == environment, "task_id"].unique()
        )
        == set(manifest["final_test"]["task_ids"]),
        "exported_states_are_final_only": set(
            action.loc[
                action["environment"] == environment, "state_id"
            ].unique()
        )
        == set(manifest["final_test"]["state_ids"]),
    }

truth_invariance_group = [
    "environment",
    "state_id",
    "task_id",
    "horizon",
    "action",
]
truth_cost_nunique = action.groupby(truth_invariance_group)[
    "true_cost"
].nunique(dropna=False)
interaction_nunique = action.groupby(truth_invariance_group)[
    "interaction_type"
].nunique(dropna=False)
truth_invariance = {
    "true_cost_max_unique_values": int(truth_cost_nunique.max()),
    "interaction_type_max_unique_values": int(interaction_nunique.max()),
}

controls = {}
blind = action[action["readout"] == "action_blind"]
oracle = action[action["readout"] == "oracle_pose"]
controls["action_blind_max_abs_predicted_cost"] = float(
    blind["predicted_cost"].abs().max()
)
controls["oracle_max_abs_cost_error"] = float(
    (oracle["predicted_cost"] - oracle["true_cost"]).abs().max()
)
linear = action[action["readout"] == "linear_pose"].copy()
shuffled = action[action["readout"] == "linear_pose_shuffled"].copy()
shift_key = [
    "environment",
    "state_id",
    "task_id",
    "model",
    "probe_seed",
    "horizon",
    "action",
]
linear_shift = linear[shift_key + ["predicted_cost"]].copy()
linear_shift["action"] = (linear_shift["action"] + 1) % 10
linear_shift = linear_shift.rename(
    columns={"predicted_cost": "expected_shifted_cost"}
)
shift_join = shuffled.merge(linear_shift, on=shift_key, validate="one_to_one")
controls["shuffled_max_abs_roll_error"] = float(
    (
        shift_join["predicted_cost"]
        - shift_join["expected_shifted_cost"]
    )
    .abs()
    .max()
)

units = (
    action.groupby(action_group, sort=True, observed=True)
    .apply(unit_metrics)
    .reset_index()
)

point_metrics = [
    "pose_error",
    "physical_cost_rmse",
    "top1_correct",
    "normalized_regret",
    "pairwise_accuracy",
    "weighted_pairwise_accuracy",
    "normalized_margin_rmse",
]
recomputed_summary = (
    units.groupby(summary_key, observed=True)[point_metrics]
    .mean()
    .reset_index()
)
summary_join = summary.merge(
    recomputed_summary,
    on=summary_key,
    suffixes=("_reported", "_recomputed"),
    validate="one_to_one",
)
summary_differences = {}
for metric in point_metrics:
    difference = (
        summary_join[f"{metric}_reported"]
        - summary_join[f"{metric}_recomputed"]
    ).abs()
    summary_differences[metric] = (
        float(difference.max(skipna=True))
        if difference.notna().any()
        else np.nan
    )

action_lookup = action.set_index(action_key)
pair_join = pair.merge(
    action[
        action_key
        + ["true_cost", "predicted_cost", "interaction_type"]
    ].rename(
        columns={
            "action": "pair_left",
            "true_cost": "left_true_cost",
            "predicted_cost": "left_predicted_cost",
            "interaction_type": "expected_left_interaction_type",
        }
    ),
    on=action_group + ["pair_left"],
    validate="many_to_one",
).merge(
    action[
        action_key
        + ["true_cost", "predicted_cost", "interaction_type"]
    ].rename(
        columns={
            "action": "pair_right",
            "true_cost": "right_true_cost",
            "predicted_cost": "right_predicted_cost",
            "interaction_type": "expected_right_interaction_type",
        }
    ),
    on=action_group + ["pair_right"],
    validate="many_to_one",
)
expected_true_margin = (
    pair_join["left_true_cost"] - pair_join["right_true_cost"]
)
expected_predicted_margin = (
    pair_join["left_predicted_cost"] - pair_join["right_predicted_cost"]
)
valid = expected_true_margin.abs() > TIE
expected_credit = np.full(len(pair_join), np.nan)
same = np.sign(expected_true_margin) == np.sign(expected_predicted_margin)
expected_credit[valid & same] = 1.0
expected_credit[valid & (expected_predicted_margin.abs() <= TIE)] = 0.5
expected_credit[valid & np.isnan(expected_credit)] = 0.0
interaction_count = (
    pair_join["expected_left_interaction_type"].isin(["contact", "collision"])
).astype(int) + (
    pair_join["expected_right_interaction_type"].isin(["contact", "collision"])
).astype(int)
expected_stratum = np.array(["neither", "one", "both"])[interaction_count]
protocol_interaction_count = (
    pair_join["expected_left_interaction_type"].isin(
        ["contact", "collision", "door_cross"]
    )
).astype(int) + (
    pair_join["expected_right_interaction_type"].isin(
        ["contact", "collision", "door_cross"]
    )
).astype(int)
protocol_expected_stratum = np.array(["neither", "one", "both"])[
    protocol_interaction_count
]
pair_reconciliation = {
    "pair_order_all_valid": bool(
        (pair_join["pair_left"] < pair_join["pair_right"]).all()
    ),
    "true_margin_max_abs_error": float(
        (pair_join["true_margin"] - expected_true_margin).abs().max()
    ),
    "predicted_margin_max_abs_error": float(
        (
            pair_join["predicted_margin"] - expected_predicted_margin
        ).abs().max()
    ),
    "margin_weight_max_abs_error": float(
        (pair_join["margin_weight"] - expected_true_margin.abs()).abs().max()
    ),
    "ranking_credit_mismatches": int(
        np.sum(
            ~(
                np.isclose(
                    pair_join["ranking_credit"],
                    expected_credit,
                    equal_nan=True,
                )
            )
        )
    ),
    "left_interaction_type_mismatches": int(
        (
            pair_join["left_interaction_type"]
            != pair_join["expected_left_interaction_type"]
        ).sum()
    ),
    "right_interaction_type_mismatches": int(
        (
            pair_join["right_interaction_type"]
            != pair_join["expected_right_interaction_type"]
        ).sum()
    ),
    "interaction_stratum_mismatches_under_collision_definition": int(
        (pair_join["interaction_stratum"].to_numpy() != expected_stratum).sum()
    ),
    "interaction_stratum_mismatches_under_written_protocol": int(
        (
            pair_join["interaction_stratum"].to_numpy()
            != protocol_expected_stratum
        ).sum()
    ),
}

decision_recomputed = {}
for environment in ["PushT", "Wall", "pooled"]:
    selected_units = units[
        (units["environment"] == environment)
        if environment != "pooled"
        else np.ones(len(units), dtype=bool)
    ]
    baseline = selected_units[
        selected_units["readout"] == "latent_distance"
    ].set_index(
        ["environment", "state_id", "model", "horizon", "probe_seed"]
    )
    candidate = selected_units[
        selected_units["readout"] == "linear_pose"
    ].set_index(
        ["environment", "state_id", "model", "horizon", "probe_seed"]
    )
    common = baseline.index.intersection(candidate.index).sort_values()
    groups = np.asarray([f"{key[0]}:{key[1]}" for key in common])
    seed_offset = (
        0 if environment == "pooled" else 100 * ["PushT", "Wall"].index(environment)
    )
    seed = SEED + 5000 + seed_offset + READOUTS.index("linear_pose")
    metrics = {
        "normalized_regret_improvement": (
            baseline.loc[common, "normalized_regret"].to_numpy()
            - candidate.loc[common, "normalized_regret"].to_numpy()
        ),
        "weighted_pairwise_accuracy_improvement": (
            candidate.loc[common, "weighted_pairwise_accuracy"].to_numpy()
            - baseline.loc[common, "weighted_pairwise_accuracy"].to_numpy()
        ),
        "top1_accuracy_improvement": (
            candidate.loc[common, "top1_correct"].to_numpy()
            - baseline.loc[common, "top1_correct"].to_numpy()
        ),
    }
    decision_recomputed[environment] = {
        name: bootstrap_mean(values, groups, BOOTSTRAP_REPS, seed)
        for name, values in metrics.items()
    }

decision_interval_matches = {
    environment: {
        metric: interval_close(
            decision_recomputed[environment][metric],
            decision["environment_comparisons"][environment][metric],
        )
        for metric in decision_recomputed[environment]
    }
    for environment in decision_recomputed
}

control_comparisons = {}
for environment in ["PushT", "Wall"]:
    env_units = units[units["environment"] == environment]
    candidate = env_units[env_units["readout"] == "linear_pose"].set_index(
        ["environment", "state_id", "model", "horizon", "probe_seed"]
    )
    control_comparisons[environment] = {}
    for baseline_name in [
        "latent_distance",
        "action_blind",
        "linear_pose_shuffled",
    ]:
        baseline = env_units[
            env_units["readout"] == baseline_name
        ].set_index(
            ["environment", "state_id", "model", "horizon", "probe_seed"]
        )
        common = baseline.index.intersection(candidate.index).sort_values()
        groups = np.asarray([f"{key[0]}:{key[1]}" for key in common])
        seed = (
            SEED
            + 6000
            + 100 * ["PushT", "Wall"].index(environment)
            + READOUTS.index(baseline_name)
        )
        control_comparisons[environment][baseline_name] = {
            "normalized_regret_improvement": bootstrap_mean(
                baseline.loc[common, "normalized_regret"].to_numpy()
                - candidate.loc[common, "normalized_regret"].to_numpy(),
                groups,
                BOOTSTRAP_REPS,
                seed,
            ),
            "weighted_pairwise_accuracy_improvement": bootstrap_mean(
                candidate.loc[
                    common, "weighted_pairwise_accuracy"
                ].to_numpy()
                - baseline.loc[
                    common, "weighted_pairwise_accuracy"
                ].to_numpy(),
                groups,
                BOOTSTRAP_REPS,
                seed,
            ),
        }

degenerate = (
    units.groupby(["environment", "state_id", "horizon"])[
        "true_cost_spread"
    ]
    .first()
    .reset_index()
)
degenerate = degenerate[degenerate["true_cost_spread"] <= TIE]
undefined_margin = (
    units.groupby(["environment", "horizon", "readout"])[
        "normalized_margin_rmse"
    ]
    .agg(total="size", undefined=lambda values: int(values.isna().sum()))
    .reset_index()
)
undefined_margin = undefined_margin[undefined_margin["undefined"] > 0]

candidate_reconciliation = {}
for environment, filename in [
    ("PushT", "pusht_design.npz"),
    ("Wall", "wall_design.npz"),
]:
    with np.load(BUNDLE / filename, allow_pickle=True) as design:
        costs = design["physical_cost"].astype(float)
        interactions = design["interactions"]
        interaction_types = design["interaction_types"].astype(str)
        labels = design["candidate_labels"].astype(str).tolist()
        left, right = np.triu_indices(costs.shape[1], k=1)
        strata_counts = {"neither": 0, "one": 0, "both": 0}
        for horizon_index in range(costs.shape[2]):
            count = (
                (interactions[:, left, horizon_index] > 0).astype(int)
                + (interactions[:, right, horizon_index] > 0).astype(int)
            )
            for index, name in enumerate(["neither", "one", "both"]):
                strata_counts[name] += int(np.sum(count == index))
        no_op_oracle = [
            float(np.mean(np.argmin(costs[:, :, index], axis=1) == 0))
            for index in range(costs.shape[2])
        ]
        no_op_positive_regret = [
            float(
                np.mean(
                    costs[:, 0, index]
                    > np.min(costs[:, :, index], axis=1) + TIE
                )
            )
            for index in range(costs.shape[2])
        ]
        median_spread = [
            float(
                np.median(
                    np.max(costs[:, :, index], axis=1)
                    - np.min(costs[:, :, index], axis=1)
                )
            )
            for index in range(costs.shape[2])
        ]
        minimum_spread = [
            float(
                np.min(
                    np.max(costs[:, :, index], axis=1)
                    - np.min(costs[:, :, index], axis=1)
                )
            )
            for index in range(costs.shape[2])
        ]
        interaction_fraction = [
            float(np.mean(interactions[:, :, index] > 0))
            for index in range(costs.shape[2])
        ]
        reported = candidate_summary[environment]
        candidate_reconciliation[environment] = {
            "labels_match": labels == reported["candidate_labels"],
            "no_op_oracle_max_abs_error": float(
                np.max(
                    np.abs(
                        np.asarray(no_op_oracle)
                        - np.asarray(
                            reported["no_op_oracle_fraction_by_horizon"]
                        )
                    )
                )
            ),
            "no_op_positive_regret_max_abs_error": float(
                np.max(
                    np.abs(
                        np.asarray(no_op_positive_regret)
                        - np.asarray(
                            reported[
                                "no_op_positive_regret_fraction_by_horizon"
                            ]
                        )
                    )
                )
            ),
            "median_spread_max_abs_error": float(
                np.max(
                    np.abs(
                        np.asarray(median_spread)
                        - np.asarray(
                            reported[
                                "median_physical_cost_spread_by_horizon"
                            ]
                        )
                    )
                )
            ),
            "minimum_spread_max_abs_error": float(
                np.max(
                    np.abs(
                        np.asarray(minimum_spread)
                        - np.asarray(
                            reported[
                                "minimum_physical_cost_spread_by_horizon"
                            ]
                        )
                    )
                )
            ),
            "interaction_fraction_max_abs_error": float(
                np.max(
                    np.abs(
                        np.asarray(interaction_fraction)
                        - np.asarray(
                            reported["interaction_fraction_by_horizon"]
                        )
                    )
                )
            ),
            "strata_counts_match": strata_counts
            == reported["pair_interaction_counts"],
            "protocol_interaction_fraction_by_horizon": [
                float(
                    np.mean(
                        np.isin(
                            interaction_types[:, :, index],
                            ["contact", "collision", "door_cross"],
                        )
                    )
                )
                for index in range(costs.shape[2])
            ],
            "protocol_strata_counts": {
                name: int(
                    sum(
                        np.sum(
                            (
                                np.isin(
                                    interaction_types[:, left, index],
                                    ["contact", "collision", "door_cross"],
                                ).astype(int)
                                + np.isin(
                                    interaction_types[:, right, index],
                                    ["contact", "collision", "door_cross"],
                                ).astype(int)
                            )
                            == stratum_index
                        )
                        for index in range(costs.shape[2])
                    )
                )
                for stratum_index, name in enumerate(
                    ["neither", "one", "both"]
                )
            },
            "reported_design_valid": reported["design_valid"],
        }

probe_files = sorted((BUNDLE / "probes").glob("*.npz"))
probe_file_checks = []
for path in probe_files:
    with np.load(path, allow_pickle=True) as probe:
        probe_file_checks.append(
            {
                "name": path.name,
                "keys": sorted(probe.files),
                "all_numeric_finite": bool(
                    all(
                        np.isfinite(probe[key]).all()
                        for key in probe.files
                        if np.issubdtype(probe[key].dtype, np.number)
                    )
                ),
                "coefficient_shape": list(probe["coefficient"].shape),
            }
        )

interaction_summary = (
    pair[pair["readout"].isin(["latent_distance", "linear_pose"])]
    .assign(weighted_credit=lambda frame: frame["ranking_credit"] * frame["margin_weight"])
    .groupby(["environment", "readout", "interaction_stratum"], observed=True)
    .agg(
        rows=("ranking_credit", "size"),
        valid_pairs=("ranking_credit", "count"),
        weight=("margin_weight", "sum"),
        weighted_credit=("weighted_credit", "sum"),
    )
    .reset_index()
)
interaction_summary["weighted_pairwise_accuracy"] = (
    interaction_summary["weighted_credit"] / interaction_summary["weight"]
)

pair_join["protocol_interaction_stratum"] = protocol_expected_stratum
interaction_summary_protocol = (
    pair_join[
        pair_join["readout"].isin(["latent_distance", "linear_pose"])
    ]
    .assign(
        weighted_credit=lambda frame: frame["ranking_credit"]
        * frame["margin_weight"]
    )
    .groupby(
        ["environment", "readout", "protocol_interaction_stratum"],
        observed=True,
    )
    .agg(
        rows=("ranking_credit", "size"),
        valid_pairs=("ranking_credit", "count"),
        weight=("margin_weight", "sum"),
        weighted_credit=("weighted_credit", "sum"),
    )
    .reset_index()
)
interaction_summary_protocol["weighted_pairwise_accuracy"] = (
    interaction_summary_protocol["weighted_credit"]
    / interaction_summary_protocol["weight"]
)

planning_summary = (
    units[units["readout"].isin(["latent_distance", "linear_pose"])]
    .groupby(["environment", "readout", "horizon"], observed=True)[
        ["top1_correct", "normalized_regret", "weighted_pairwise_accuracy"]
    ]
    .mean()
    .reset_index()
)

model_ranking_issues = {
    "rows_with_nonfinite_counterfactual_metric": int(
        (
            ~np.isfinite(rankings["counterfactual_margin_rmse"])
            | ~np.isfinite(rankings["weighted_pairwise_accuracy"])
        ).sum()
    ),
    "reported_reversal_rows_with_nonfinite_metric": int(
        (
            rankings["ordinary_vs_counterfactual_reversal"]
            & (
                ~np.isfinite(rankings["counterfactual_margin_rmse"])
                | ~np.isfinite(rankings["weighted_pairwise_accuracy"])
            )
        ).sum()
    ),
    "ordinary_vs_planning_rank_disagreements": int(
        (rankings["ordinary_rank"] != rankings["planning_rank"]).sum()
    ),
}

exploratory_final_associations = {}
for environment in ["PushT", "Wall", "pooled"]:
    selected = units[
        (units["readout"] == "linear_pose")
        & (
            (units["environment"] == environment)
            if environment != "pooled"
            else np.ones(len(units), dtype=bool)
        )
    ]
    exploratory_final_associations[environment] = {}
    for predictor in [
        "pose_error",
        "physical_cost_rmse",
        "normalized_margin_rmse",
    ]:
        finite = selected[
            np.isfinite(selected[predictor])
            & np.isfinite(selected["normalized_regret"])
        ]
        exploratory_final_associations[environment][predictor] = {
            "n_rows": int(len(finite)),
            "pearson": float(
                finite[predictor].corr(
                    finite["normalized_regret"], method="pearson"
                )
            ),
            "spearman": float(
                finite[predictor]
                .rank(method="average")
                .corr(
                    finite["normalized_regret"].rank(method="average"),
                    method="pearson",
                )
            ),
        }

regression_nan_counts = regression_predictions.isna().sum().to_dict()
raw_regression = load_json("held_out_regression.json")

checks = {
    "row_counts_match": row_counts == expected_row_counts,
    "no_duplicate_keys": all(value == 0 for value in duplicates.values()),
    "action_groups_have_10_rows": group_sizes["actions"] == {10: 7200},
    "pair_groups_have_45_rows": group_sizes["pairs"] == {45: 7200},
    "split_integrity": all(
        item["task_sets_disjoint"]
        and item["state_sets_disjoint"]
        and item["exported_tasks_are_final_only"]
        and item["exported_states_are_final_only"]
        for item in task_split_checks.values()
    ),
    "truth_invariant": all(value == 1 for value in truth_invariance.values()),
    "controls_exact": all(value == 0.0 for value in controls.values()),
    "summary_point_metrics_match": all(
        np.isnan(value) or value < 1e-12
        for value in summary_differences.values()
    ),
    "pair_rows_reconcile": (
        pair_reconciliation["pair_order_all_valid"]
        and all(
            pair_reconciliation[key] < 1e-12
            for key in [
                "true_margin_max_abs_error",
                "predicted_margin_max_abs_error",
                "margin_weight_max_abs_error",
            ]
        )
        and all(
            pair_reconciliation[key] == 0
            for key in [
                "ranking_credit_mismatches",
                "left_interaction_type_mismatches",
                "right_interaction_type_mismatches",
                "interaction_stratum_mismatches_under_collision_definition",
            ]
        )
    ),
    "decision_intervals_match": all(
        all(environment.values())
        for environment in decision_interval_matches.values()
    ),
    "candidate_summaries_reconcile": all(
        item["labels_match"]
        and item["no_op_oracle_max_abs_error"] < 1e-12
        and item["no_op_positive_regret_max_abs_error"] < 1e-12
        and item["median_spread_max_abs_error"] < 1e-12
        and item["minimum_spread_max_abs_error"] < 1e-12
        and item["interaction_fraction_max_abs_error"] < 1e-12
        and item["strata_counts_match"]
        for item in candidate_reconciliation.values()
    ),
    "restore_exact": all(
        item["endpoint_bitwise_exact"]
        and item["initial_render_bitwise_exact"]
        and item["diagnostics_exact"]
        and item["max_endpoint_abs_diff"] == 0.0
        for item in restore.values()
    ),
    "probe_manifest_integrity": (
        probe_manifest["world_models_frozen"]
        and not probe_manifest["test_states_used_for_fitting"]
        and probe_manifest["task_disjoint"]
        and len(probe_files) == 12
        and all(item["all_numeric_finite"] for item in probe_file_checks)
    ),
}

payload = {
    "checks": checks,
    "row_counts": row_counts,
    "duplicates": duplicates,
    "group_sizes": group_sizes,
    "task_split_checks": task_split_checks,
    "truth_invariance": truth_invariance,
    "controls": controls,
    "summary_max_abs_differences": summary_differences,
    "pair_reconciliation": pair_reconciliation,
    "decision_recomputed": decision_recomputed,
    "decision_interval_matches": decision_interval_matches,
    "control_comparisons": control_comparisons,
    "candidate_reconciliation": candidate_reconciliation,
    "degenerate_final_units": degenerate.to_dict(orient="records"),
    "undefined_margin_counts": undefined_margin.to_dict(orient="records"),
    "regression_nan_counts": regression_nan_counts,
    "held_out_regression": raw_regression,
    "planning_summary": planning_summary.to_dict(orient="records"),
    "interaction_summary": interaction_summary.to_dict(orient="records"),
    "interaction_summary_under_written_protocol": (
        interaction_summary_protocol.to_dict(orient="records")
    ),
    "model_ranking_issues": model_ranking_issues,
    "exploratory_final_associations": exploratory_final_associations,
    "probe_file_checks": probe_file_checks,
}
print(json.dumps(payload, indent=2, allow_nan=True))
