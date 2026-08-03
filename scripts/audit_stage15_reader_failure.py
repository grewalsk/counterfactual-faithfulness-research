#!/usr/bin/env python3
"""Offline, post-hoc audit of the Stage 15 fixed-reader gate failure.

This script never loads JEPA or a simulator. It consumes only the frozen
``truth`` and ``target_tokens`` arrays saved before the reader gate stopped the
pilot. All hyperparameter selection is leave-one-construction-trajectory-out;
evaluation trajectories are opened only for the final descriptive comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cf_faithfulness.stage15_bundle import (  # noqa: E402
    PHYSICAL_READER_LABELS,
    physical_reader_targets,
    r2_per_output,
)


DEFAULT_RAW = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "stage15"
    / "pilot_5703d35fb8b5"
)
DEFAULT_OUTPUT = REPO / "audits" / "stage15" / "reader_failure_audit"
RIDGES = np.asarray(
    [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6],
    dtype=np.float64,
)
CONSTRUCTION_TRAJECTORIES = (0, 2, 4, 6)
EVALUATION_TRAJECTORIES = (1, 3, 5, 7)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n")


def write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root):
    manifest = json.loads((root / "full_manifest.json").read_text())
    missing = []
    mismatches = []
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.is_file():
            missing.append(row["path"])
            continue
        observed = sha256_file(path)
        if observed != row["sha256"]:
            mismatches.append(
                {
                    "path": row["path"],
                    "expected": row["sha256"],
                    "observed": observed,
                }
            )
    result = {
        "manifest_files": len(manifest["files"]),
        "manifest_bytes": int(sum(row["size_bytes"] for row in manifest["files"])),
        "missing": missing,
        "hash_mismatches": mismatches,
        "passed": not missing and not mismatches,
    }
    if not result["passed"]:
        raise RuntimeError(f"raw evidence verification failed: {result}")
    return result


class NumpyCountSketch:
    def __init__(self, input_dim, output_dim, seed):
        rng = np.random.default_rng(int(seed))
        bucket = rng.integers(0, output_dim, size=input_dim, dtype=np.int64)
        sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), input_dim)
        counts = np.bincount(bucket, minlength=output_dim).astype(np.float32)
        counts[counts == 0] = 1.0
        data = sign / np.sqrt(counts[bucket])
        self.matrix = sparse.csr_matrix(
            (data, (np.arange(input_dim), bucket)),
            shape=(input_dim, output_dim),
        )

    def __call__(self, values):
        result = np.asarray(values.reshape(len(values), -1) @ self.matrix)
        return result.astype(np.float32)


def coordinate_basis(degree):
    centers = (np.arange(16, dtype=np.float64) + 0.5) / 16.0
    yy, xx = np.meshgrid(centers, centers, indexing="ij")
    x = 2.0 * xx.reshape(-1) - 1.0
    y = 2.0 * yy.reshape(-1) - 1.0
    columns = [np.ones_like(x)]
    if degree >= 1:
        columns.extend([x, y])
    if degree >= 2:
        columns.extend([0.5 * (3.0 * x**2 - 1.0), x * y, 0.5 * (3.0 * y**2 - 1.0)])
    return np.stack(columns, axis=1)


def coordinate_moment_features(tokens, degree):
    basis = coordinate_basis(degree)
    values = np.einsum("npc,pb->nbc", tokens, basis, optimize=True)
    return (values / 256.0).reshape(len(tokens), -1).astype(np.float32)


def spatial_pool_features(tokens, bins=4):
    if 16 % bins:
        raise ValueError("bins must divide the 16x16 token grid")
    width = 16 // bins
    grid = tokens.reshape(len(tokens), 16, 16, tokens.shape[-1])
    pooled = grid.reshape(
        len(tokens), bins, width, bins, width, tokens.shape[-1]
    ).mean(axis=(2, 4))
    return pooled.reshape(len(tokens), -1).astype(np.float32)


def object_masks(image):
    red = image[..., 0].astype(np.int16)
    green = image[..., 1].astype(np.int16)
    blue = image[..., 2].astype(np.int16)
    agent = (blue - red > 80) & (blue - green > 30) & (blue > 180) & (green < 190)
    block = (
        (blue - red > 20)
        & (green - red > 8)
        & (blue < 220)
        & ~agent
    )
    return agent, block


def patch_fraction(mask):
    return mask.reshape(16, 14, 16, 14).mean(axis=(1, 3))


def mask_geometry(mask):
    weights = np.asarray(mask, dtype=np.float64)
    height, width = weights.shape
    y = (np.arange(height, dtype=np.float64) + 0.5) / height
    x = (np.arange(width, dtype=np.float64) + 0.5) / width
    yy, xx = np.meshgrid(y, x, indexing="ij")
    total = max(float(weights.sum()), 1e-12)
    center_x = float(np.sum(weights * xx) / total)
    center_y = float(np.sum(weights * yy) / total)
    dx = xx - center_x
    dy = yy - center_y
    moments = [
        np.sum(weights * dx**2) / total,
        np.sum(weights * dx * dy) / total,
        np.sum(weights * dy**2) / total,
        np.sum(weights * dx**3) / total,
        np.sum(weights * dx**2 * dy) / total,
        np.sum(weights * dx * dy**2) / total,
        np.sum(weights * dy**3) / total,
    ]
    return np.asarray([center_x, center_y, *moments], dtype=np.float64)


def fit_ridge(features, targets, ridge):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (x - mean) / scale
    intercept = y.mean(axis=0)
    centered = y - intercept
    if standardized.shape[1] <= standardized.shape[0]:
        coefficient = np.linalg.solve(
            standardized.T @ standardized
            + float(ridge) * np.eye(standardized.shape[1]),
            standardized.T @ centered,
        )
    else:
        dual = np.linalg.solve(
            standardized @ standardized.T
            + float(ridge) * np.eye(standardized.shape[0]),
            centered,
        )
        coefficient = standardized.T @ dual
    return {
        "mean": mean,
        "scale": scale,
        "intercept": intercept,
        "coefficient": coefficient,
        "ridge": float(ridge),
    }


def predict_ridge(model, features):
    x = np.asarray(features, dtype=np.float64)
    return (
        (x - model["mean"]) / model["scale"] @ model["coefficient"]
        + model["intercept"]
    )


def grouped_cv(features, targets, groups, ridges=RIDGES):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    group_values = np.asarray(groups)
    rows = []
    for ridge in ridges:
        losses = []
        for held_out in np.unique(group_values):
            train = group_values != held_out
            model = fit_ridge(x[train], y[train], ridge)
            prediction = predict_ridge(model, x[~train])
            loss = float(np.mean((prediction - y[~train]) ** 2))
            losses.append(loss)
            rows.append(
                {
                    "ridge": float(ridge),
                    "held_out_group": int(held_out),
                    "mse": loss,
                }
            )
        rows.append(
            {
                "ridge": float(ridge),
                "held_out_group": "mean",
                "mse": float(np.mean(losses)),
            }
        )
    mean_loss = {
        float(row["ridge"]): float(row["mse"])
        for row in rows
        if row["held_out_group"] == "mean"
    }
    selected = min((float(value) for value in ridges), key=lambda value: (mean_loss[value], value))
    return {
        "selected_ridge": selected,
        "selected_cv_mse": mean_loss[selected],
        "rows": rows,
        "model": fit_ridge(x, y, selected),
    }


def split_metrics(target, prediction):
    values = r2_per_output(target, prediction)
    return {
        "r2": dict(zip(PHYSICAL_READER_LABELS, (float(value) for value in values))),
        "median_r2": float(np.median(values)),
        "minimum_spatial_r2": float(np.min(values[:4])),
        "original_gate_passed": bool(np.median(values) >= 0.25 and np.min(values[:4]) >= 0.10),
    }


def load_features(root):
    frozen = np.load(root / "analysis" / "fixed_physical_readers.npz")
    projection_seeds = [int(value) for value in frozen["projection_seeds"]]
    sketches = {}
    for dimension in (192, 1152, 2304):
        for seed in projection_seeds:
            name = (
                f"countsketch_{seed}"
                if dimension == 192
                else f"countsketch_{dimension}_{seed}"
            )
            sketches[name] = NumpyCountSketch(256 * 384, dimension, seed)
    feature_names = [
        *sketches,
        "channel_mean",
        "coordinate_moments_degree1",
        "coordinate_moments_degree2",
        "spatial_pool_4x4",
        "oracle_patch_geometry",
    ]
    chunks = {name: [] for name in feature_names}
    targets = []
    groups = []
    splits = []
    sample_rows = []
    renderer_rows = []

    for truth_path in sorted((root / "truth").glob("state_*.npz")):
        with np.load(truth_path) as truth:
            record_id = int(truth["record_id"])
            trajectory_id = int(truth["trajectory_id"])
            time_index = int(truth["time_index"])
            split = str(truth["split"])
            endpoints = truth["endpoint_states"].astype(np.float64)
            visuals = truth["future_visual"]
        with np.load(root / "target_tokens" / f"state_{record_id:04d}.npz") as payload:
            tokens = payload["true_tokens"].astype(np.float32).reshape(-1, 256, 384)

        if tokens.shape != (26, 256, 384):
            raise RuntimeError(f"unexpected token shape for {record_id}: {tokens.shape}")
        physical = physical_reader_targets(endpoints).reshape(-1, 6)
        flat = tokens.reshape(len(tokens), -1)
        for name, projector in sketches.items():
            chunks[name].append(projector(flat))
        chunks["channel_mean"].append(coordinate_moment_features(tokens, 0))
        chunks["coordinate_moments_degree1"].append(
            coordinate_moment_features(tokens, 1)
        )
        chunks["coordinate_moments_degree2"].append(
            coordinate_moment_features(tokens, 2)
        )
        chunks["spatial_pool_4x4"].append(spatial_pool_features(tokens, bins=4))

        geometry = []
        for sample_index, (image, state) in enumerate(
            zip(visuals.reshape(-1, 224, 224, 3), endpoints.reshape(-1, 10))
        ):
            agent_mask, block_mask = object_masks(image)
            agent_pixel = mask_geometry(agent_mask)
            block_pixel = mask_geometry(block_mask)
            agent_patch = mask_geometry(patch_fraction(agent_mask))
            block_patch = mask_geometry(patch_fraction(block_mask))
            geometry.append(np.concatenate([agent_patch[:2], block_patch]))
            renderer_rows.append(
                {
                    "record_id": record_id,
                    "trajectory_id": trajectory_id,
                    "time_index": time_index,
                    "split": split,
                    "sample_index": sample_index,
                    "agent_pixel_error": float(
                        np.linalg.norm(agent_pixel[:2] - state[:2] / 512.0)
                    ),
                    "agent_patch_error": float(
                        np.linalg.norm(agent_patch[:2] - state[:2] / 512.0)
                    ),
                    "block_pixel_centroid_offset": float(
                        np.linalg.norm(block_pixel[:2] - state[2:4] / 512.0)
                    ),
                    "block_patch_centroid_offset": float(
                        np.linalg.norm(block_patch[:2] - state[2:4] / 512.0)
                    ),
                    "agent_patch_support": int(np.sum(patch_fraction(agent_mask) > 0.02)),
                    "block_patch_support": int(np.sum(patch_fraction(block_mask) > 0.02)),
                }
            )
        chunks["oracle_patch_geometry"].append(np.stack(geometry).astype(np.float32))
        targets.append(physical)
        groups.append(np.full(len(tokens), trajectory_id, dtype=np.int64))
        splits.append(np.full(len(tokens), split))
        for sample_index in range(len(tokens)):
            sample_rows.append(
                {
                    "record_id": record_id,
                    "trajectory_id": trajectory_id,
                    "time_index": time_index,
                    "split": split,
                    "action_index": sample_index // 2,
                    "horizon": (1, 3)[sample_index % 2],
                }
            )

    return {
        "features": {name: np.concatenate(values) for name, values in chunks.items()},
        "targets": np.concatenate(targets),
        "groups": np.concatenate(groups),
        "splits": np.concatenate(splits),
        "samples": sample_rows,
        "renderer_rows": renderer_rows,
        "frozen": {name: frozen[name].copy() for name in frozen.files},
    }


def reproduce_frozen_reader(dataset, saved_gate):
    arrays = dataset["frozen"]
    predictions = []
    for index, seed in enumerate(arrays["projection_seeds"]):
        features = dataset["features"][f"countsketch_{int(seed)}"]
        prediction = (
            (features - arrays["feature_mean"][index])
            / arrays["feature_scale"][index]
            @ arrays["coefficient"][index]
            + arrays["intercept"][index]
        )
        predictions.append(prediction)
    prediction = np.mean(predictions, axis=0)
    target = (
        dataset["targets"] - arrays["target_mean"]
    ) / arrays["target_scale"]
    evaluation = dataset["splits"] == "evaluation"
    metrics = split_metrics(target[evaluation], prediction[evaluation])
    expected = np.asarray([saved_gate["r2"][name] for name in PHYSICAL_READER_LABELS])
    observed = np.asarray([metrics["r2"][name] for name in PHYSICAL_READER_LABELS])
    return {
        **metrics,
        "expected_r2": dict(zip(PHYSICAL_READER_LABELS, expected.tolist())),
        "max_abs_r2_reproduction_error": float(np.max(np.abs(expected - observed))),
        "passed": bool(np.max(np.abs(expected - observed)) <= 1e-6),
    }


def evaluate_feature_model(
    name,
    features,
    target,
    groups,
    horizons,
    construction,
    evaluation,
):
    cv = grouped_cv(features[construction], target[construction], groups[construction])
    construction_prediction = predict_ridge(cv["model"], features[construction])
    evaluation_prediction = predict_ridge(cv["model"], features[evaluation])
    construction_groups = groups[construction]
    oof_prediction = np.empty_like(target[construction])
    for held_out in np.unique(construction_groups):
        train = construction_groups != held_out
        fold_model = fit_ridge(
            features[construction][train],
            target[construction][train],
            cv["selected_ridge"],
        )
        oof_prediction[~train] = predict_ridge(
            fold_model, features[construction][~train]
        )
    evaluation_groups = groups[evaluation]
    evaluation_horizons = horizons[evaluation]
    by_trajectory = {}
    for trajectory_id in np.unique(evaluation_groups):
        selected = evaluation_groups == trajectory_id
        by_trajectory[str(int(trajectory_id))] = split_metrics(
            target[evaluation][selected], evaluation_prediction[selected]
        )
    by_horizon = {}
    for horizon in np.unique(evaluation_horizons):
        selected = evaluation_horizons == horizon
        by_horizon[str(int(horizon))] = split_metrics(
            target[evaluation][selected], evaluation_prediction[selected]
        )
    leave_one_trajectory_out = {}
    for excluded in np.unique(evaluation_groups):
        selected = evaluation_groups != excluded
        leave_one_trajectory_out[str(int(excluded))] = split_metrics(
            target[evaluation][selected], evaluation_prediction[selected]
        )
    return {
        "name": name,
        "feature_dimension": int(features.shape[1]),
        "selected_ridge": cv["selected_ridge"],
        "construction_cv_mse": cv["selected_cv_mse"],
        "construction": split_metrics(target[construction], construction_prediction),
        "construction_oof": split_metrics(target[construction], oof_prediction),
        "evaluation_post_hoc": split_metrics(target[evaluation], evaluation_prediction),
        "evaluation_post_hoc_by_trajectory": by_trajectory,
        "evaluation_post_hoc_by_horizon": by_horizon,
        "evaluation_post_hoc_leave_one_trajectory_out": leave_one_trajectory_out,
        "cv_rows": cv["rows"],
        "model": cv["model"],
        "_construction_oof_prediction": oof_prediction,
    }


def plot_metrics(output, results, frozen):
    selected_names = [
        "frozen_countsketch_192",
        "countsketch_192_extended",
        "countsketch_1152_extended",
        "countsketch_2304_extended",
        "channel_mean",
        "coordinate_moments_degree1",
        "coordinate_moments_degree2",
        "spatial_pool_4x4",
    ]
    values = []
    for name in selected_names:
        source = frozen if name == "frozen_countsketch_192" else results[name]["evaluation_post_hoc"]
        values.append([source["r2"][label] for label in PHYSICAL_READER_LABELS])
    matrix = np.asarray(values)
    figure, axis = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(PHYSICAL_READER_LABELS))
    width = 0.095
    for index, (name, row) in enumerate(zip(selected_names, matrix)):
        axis.bar(x + (index - 3.5) * width, row, width=width, label=name)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axhline(0.10, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    axis.set_xticks(x)
    axis.set_xticklabels(PHYSICAL_READER_LABELS, rotation=25, ha="right")
    axis.set_ylabel("Evaluation R² (post-hoc except frozen reader)")
    axis.set_title("Stage 15 fixed-reader failure audit")
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output / "reader_model_comparison.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    verification = verify_manifest(root)
    dataset = load_features(root)
    construction = dataset["splits"] == "construction"
    evaluation = dataset["splits"] == "evaluation"
    if int(construction.sum()) != 520 or int(evaluation.sum()) != 520:
        raise RuntimeError("expected 520 construction and 520 evaluation examples")
    arrays = dataset["frozen"]
    target = (
        dataset["targets"] - arrays["target_mean"]
    ) / arrays["target_scale"]
    saved_gate = json.loads(
        (root / "analysis" / "reader_evaluation_gate.json").read_text()
    )
    reproduction = reproduce_frozen_reader(dataset, saved_gate)
    if not reproduction["passed"]:
        raise RuntimeError(f"frozen reader did not reproduce: {reproduction}")

    results = {}
    horizons = np.asarray([row["horizon"] for row in dataset["samples"]])
    for dimension in (192, 1152, 2304):
        individual_countsketch = []
        for seed in arrays["projection_seeds"]:
            name = (
                f"countsketch_{int(seed)}"
                if dimension == 192
                else f"countsketch_{dimension}_{int(seed)}"
            )
            result = evaluate_feature_model(
                name,
                dataset["features"][name],
                target,
                dataset["groups"],
                horizons,
                construction,
                evaluation,
            )
            results[name] = result
            individual_countsketch.append(result)

        construction_predictions = []
        evaluation_predictions = []
        for result in individual_countsketch:
            feature = dataset["features"][result["name"]]
            construction_predictions.append(
                predict_ridge(result["model"], feature[construction])
            )
            evaluation_predictions.append(
                predict_ridge(result["model"], feature[evaluation])
            )
        ensemble_name = f"countsketch_{dimension}_extended"
        results[ensemble_name] = {
            "name": ensemble_name,
            "feature_dimension": dimension,
            "selected_ridges": [
                row["selected_ridge"] for row in individual_countsketch
            ],
            "construction_cv_mse": float(
                np.mean(
                    [row["construction_cv_mse"] for row in individual_countsketch]
                )
            ),
            "construction": split_metrics(
                target[construction], np.mean(construction_predictions, axis=0)
            ),
            "construction_oof": split_metrics(
                target[construction],
                np.mean(
                    [
                        row["_construction_oof_prediction"]
                        for row in individual_countsketch
                    ],
                    axis=0,
                ),
            ),
            "evaluation_post_hoc": split_metrics(
                target[evaluation], np.mean(evaluation_predictions, axis=0)
            ),
        }

    for name in [
        "channel_mean",
        "coordinate_moments_degree1",
        "coordinate_moments_degree2",
        "spatial_pool_4x4",
        "oracle_patch_geometry",
    ]:
        results[name] = evaluate_feature_model(
            name,
            dataset["features"][name],
            target,
            dataset["groups"],
            horizons,
            construction,
            evaluation,
        )

    renderer_rows = dataset["renderer_rows"]
    renderer_summary = {}
    for key in [
        "agent_pixel_error",
        "agent_patch_error",
        "block_pixel_centroid_offset",
        "block_patch_centroid_offset",
        "agent_patch_support",
        "block_patch_support",
    ]:
        values = np.asarray([row[key] for row in renderer_rows], dtype=np.float64)
        renderer_summary[key] = {
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "maximum": float(np.max(values)),
        }

    compact_results = {}
    metric_rows = []
    cv_rows = []
    for name, result in results.items():
        compact = {
            key: value
            for key, value in result.items()
            if key not in {"model", "cv_rows"} and not key.startswith("_")
        }
        compact_results[name] = compact
        if "cv_rows" in result:
            cv_rows.extend({"model": name, **row} for row in result["cv_rows"])
        for split_name in ["construction", "evaluation_post_hoc"]:
            if split_name not in result:
                continue
            for label, value in result[split_name]["r2"].items():
                metric_rows.append(
                    {
                        "model": name,
                        "split": split_name,
                        "reader": label,
                        "r2": value,
                    }
                )

    best_post_hoc_token_reader = max(
        (
            (name, row["evaluation_post_hoc"]["median_r2"])
            for name, row in compact_results.items()
            if "evaluation_post_hoc" in row and name != "oracle_patch_geometry"
        ),
        key=lambda item: item[1],
    )
    payload = {
        "status": "POST_HOC_OFFLINE_READER_FAILURE_AUDIT",
        "claim_boundary": {
            "evaluation_used_for_model_selection": False,
            "evaluation_results_are_post_hoc": True,
            "confirmatory_claim_authorized": False,
            "operator_or_causal_claim_authorized": False,
        },
        "raw_evidence": {"root": root, **verification},
        "sample_counts": {
            "construction": int(construction.sum()),
            "evaluation": int(evaluation.sum()),
            "construction_trajectories": list(CONSTRUCTION_TRAJECTORIES),
            "evaluation_trajectories": list(EVALUATION_TRAJECTORIES),
        },
        "frozen_reader_reproduction": reproduction,
        "renderer_to_token_geometry": renderer_summary,
        "models": compact_results,
        "descriptive_best_evaluation_token_reader": {
            "name": best_post_hoc_token_reader[0],
            "median_r2": best_post_hoc_token_reader[1],
            "not_for_selection": True,
        },
    }
    write_json(output / "reader_failure_audit.json", payload)
    write_csv(output / "model_metrics.csv", metric_rows)
    write_csv(output / "construction_cv.csv", cv_rows)
    write_csv(output / "renderer_geometry.csv", renderer_rows)
    plot_metrics(output, compact_results, reproduction)
    print(json.dumps(json_safe(payload), indent=2))


if __name__ == "__main__":
    main()
