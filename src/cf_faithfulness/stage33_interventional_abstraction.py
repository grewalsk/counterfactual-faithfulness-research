"""NumPy-only numerical core for Stage 33 interventional abstraction.

The module deliberately separates three claims which are easy to conflate:

* a finite bank of interventional predictions has low numerical rank;
* fitted input/state operators agree after one fixed change of coordinates;
* an *internal* intervention is actually transported to the intended physical
  counterfactual.

The first two statements do not imply the third.  The interchange routines are
therefore independent of the rank and conjugacy routines, and the final gate
requires all of them.  All conventions use column-state dynamics

``q_next = c + A q + B a + sum_j a[j] N[j] q``.

Batch arrays store samples in rows.  A fixed cross-model map is represented as
``q_target = S q_source + offset``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _finite_array(values: ArrayLike, name: str, *, ndim: int | None = None) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _ordered_unique(values: NDArray[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[Any] = set()
    for value in values.tolist():
        key = value.item() if isinstance(value, np.generic) else value
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


# ---------------------------------------------------------------------------
# Finite-bank predictive pseudometric and rank
# ---------------------------------------------------------------------------


def bounded_predictive_signature_pseudometric(
    left: ArrayLike,
    right: ArrayLike,
    *,
    scales: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    bound: float = 1.0,
) -> float:
    """Distance between two finite interventional predictive signatures.

    This is a pseudometric on latent histories because distinct histories can
    have the same finite-bank signature.  ``min(bound, weighted_l2)`` preserves
    the triangle inequality.  It does *not* turn an epsilon-neighbour relation
    into an equivalence relation; epsilon closeness is generally nontransitive.
    """

    first = _finite_array(left, "left").reshape(-1)
    second = _finite_array(right, "right").reshape(-1)
    if first.shape != second.shape:
        raise ValueError("left and right signatures must have the same shape")
    limit = float(bound)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("bound must be positive and finite")
    if scales is None:
        scale = np.ones_like(first)
    else:
        scale = _finite_array(scales, "scales").reshape(-1)
        if scale.shape != first.shape or np.any(scale <= 0):
            raise ValueError("scales must be positive and match the signatures")
    if weights is None:
        weight = np.ones_like(first)
    else:
        weight = _finite_array(weights, "weights").reshape(-1)
        if weight.shape != first.shape or np.any(weight < 0):
            raise ValueError("weights must be nonnegative and match the signatures")
        if not np.any(weight > 0):
            raise ValueError("at least one weight must be positive")
    squared = np.sum(weight * ((first - second) / scale) ** 2) / np.sum(weight)
    return float(min(limit, np.sqrt(max(float(squared), 0.0))))


# Concise alias used by the notebook-facing code.
predictive_signature_distance = bounded_predictive_signature_pseudometric


def pairwise_predictive_signature_distances(
    signatures: ArrayLike,
    **metric_kwargs: Any,
) -> FloatArray:
    """Return the symmetric pairwise finite-bank pseudometric matrix."""

    matrix = _finite_array(signatures, "signatures", ndim=2)
    result = np.zeros((len(matrix), len(matrix)), dtype=np.float64)
    for row in range(len(matrix)):
        for column in range(row):
            value = bounded_predictive_signature_pseudometric(
                matrix[row], matrix[column], **metric_kwargs
            )
            result[row, column] = result[column, row] = value
    return result


def epsilon_predictively_related(
    left: ArrayLike,
    right: ArrayLike,
    epsilon: float,
    **metric_kwargs: Any,
) -> bool:
    """Return finite-bank epsilon closeness (a tolerance relation, not an equivalence)."""

    threshold = float(epsilon)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("epsilon must be nonnegative and finite")
    return bool(
        bounded_predictive_signature_pseudometric(left, right, **metric_kwargs)
        <= threshold
    )


def effective_rank_from_singular_values(
    singular_values: ArrayLike,
    *,
    method: str = "entropy",
) -> float:
    """Return entropy effective rank or stable rank from singular values."""

    values = np.asarray(singular_values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("singular_values must be finite, nonnegative, and nonempty")
    energy = values**2
    total = float(np.sum(energy))
    if total <= np.finfo(np.float64).tiny:
        return 0.0
    if method == "stable":
        return float(total / np.max(energy))
    if method != "entropy":
        raise ValueError("method must be 'entropy' or 'stable'")
    probabilities = energy[energy > 0] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def effective_rank(matrix: ArrayLike, *, method: str = "entropy", center: bool = False) -> float:
    """Effective rank of a matrix using its singular-value energy spectrum."""

    values = _finite_array(matrix, "matrix", ndim=2)
    if center:
        values = values - np.mean(values, axis=0, keepdims=True)
    singular_values = np.linalg.svd(values, compute_uv=False)
    return effective_rank_from_singular_values(singular_values, method=method)


def numerical_rank(
    matrix: ArrayLike,
    *,
    relative_tolerance: float | None = None,
    center: bool = False,
) -> int:
    """Hard SVD rank with an explicit relative threshold when supplied."""

    values = _finite_array(matrix, "matrix", ndim=2)
    if center:
        values = values - np.mean(values, axis=0, keepdims=True)
    singular_values = np.linalg.svd(values, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        return 0
    if relative_tolerance is None:
        tolerance = max(values.shape) * np.finfo(np.float64).eps * singular_values[0]
    else:
        relative = float(relative_tolerance)
        if not np.isfinite(relative) or relative < 0:
            raise ValueError("relative_tolerance must be nonnegative and finite")
        tolerance = relative * singular_values[0]
    return int(np.sum(singular_values > tolerance))


def _validate_cluster_ids(
    n_rows: int,
    trajectory_ids: ArrayLike,
    state_group_ids: ArrayLike | None,
) -> tuple[NDArray[Any], NDArray[Any] | None]:
    trajectories = np.asarray(trajectory_ids).reshape(-1)
    if len(trajectories) != n_rows:
        raise ValueError("trajectory_ids must have one value per row")
    if state_group_ids is None:
        states = None
    else:
        states = np.asarray(state_group_ids).reshape(-1)
        if len(states) != n_rows:
            raise ValueError("state_group_ids must have one value per row")
        for trajectory in _ordered_unique(trajectories):
            selected = trajectories == trajectory
            if len(_ordered_unique(states[selected])) != 1:
                raise ValueError("each trajectory must belong to exactly one state group")
    return trajectories, states


def _one_hierarchical_bootstrap(
    trajectories: NDArray[Any],
    states: NDArray[Any] | None,
    rng: np.random.Generator,
) -> IntArray:
    if states is None:
        units = _ordered_unique(trajectories)
        sampled = rng.integers(0, len(units), size=len(units))
        pieces = [np.flatnonzero(trajectories == units[index]) for index in sampled]
        return np.concatenate(pieces).astype(np.int64, copy=False)

    state_units = _ordered_unique(states)
    sampled_states = rng.integers(0, len(state_units), size=len(state_units))
    pieces: list[IntArray] = []
    for state_index in sampled_states:
        state = state_units[state_index]
        selected_state = states == state
        local_trajectories = _ordered_unique(trajectories[selected_state])
        sampled_trajectories = rng.integers(
            0, len(local_trajectories), size=len(local_trajectories)
        )
        for trajectory_index in sampled_trajectories:
            trajectory = local_trajectories[trajectory_index]
            pieces.append(np.flatnonzero(selected_state & (trajectories == trajectory)))
    return np.concatenate(pieces).astype(np.int64, copy=False)


def block_bootstrap_indices(
    trajectory_ids: ArrayLike,
    *,
    state_group_ids: ArrayLike | None = None,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> tuple[IntArray, ...]:
    """Resample whole trajectories, optionally hierarchically within state groups."""

    trajectories = np.asarray(trajectory_ids).reshape(-1)
    trajectories, states = _validate_cluster_ids(
        len(trajectories), trajectories, state_group_ids
    )
    repetitions = int(n_bootstrap)
    if repetitions < 1:
        raise ValueError("n_bootstrap must be positive")
    top_units = _ordered_unique(trajectories if states is None else states)
    if len(top_units) < 2:
        raise ValueError("at least two independent top-level clusters are required")
    rng = np.random.default_rng(seed)
    return tuple(
        _one_hierarchical_bootstrap(trajectories, states, rng)
        for _ in range(repetitions)
    )


def _trajectory_blocks(
    trajectories: NDArray[Any], states: NDArray[Any] | None
) -> list[tuple[Any, Any | None, IntArray]]:
    blocks: list[tuple[Any, Any | None, IntArray]] = []
    for trajectory in _ordered_unique(trajectories):
        indices = np.flatnonzero(trajectories == trajectory).astype(np.int64)
        state = None if states is None else _ordered_unique(states[indices])[0]
        blocks.append((trajectory, state, indices))
    return blocks


def structured_permute_columns(
    matrix: ArrayLike,
    trajectory_ids: ArrayLike,
    *,
    state_group_ids: ArrayLike | None = None,
    within_state_groups: bool = True,
    seed: int = 0,
) -> FloatArray:
    """Independently permute feature columns in whole equal-length blocks.

    Whole trajectories are exchanged only with equal-length trajectories and,
    by default, within the same state stratum.  If that would make every block
    a singleton, equal-length blocks are exchanged across strata.  A remaining
    singleton trajectory is circularly shifted, preserving its marginal and
    local sequence rather than applying an iid row shuffle.
    """

    values = _finite_array(matrix, "matrix", ndim=2)
    trajectories, states = _validate_cluster_ids(
        len(values), trajectory_ids, state_group_ids
    )
    blocks = _trajectory_blocks(trajectories, states)

    def make_buckets(use_states: bool) -> dict[tuple[Any, int], list[IntArray]]:
        result: dict[tuple[Any, int], list[IntArray]] = {}
        for _, state, indices in blocks:
            key = (state if use_states else None, len(indices))
            result.setdefault(key, []).append(indices)
        return result

    buckets = make_buckets(bool(within_state_groups and states is not None))
    if len(blocks) > 1 and all(len(bucket) == 1 for bucket in buckets.values()):
        buckets = make_buckets(False)
    rng = np.random.default_rng(seed)
    permuted = np.empty_like(values)
    for column in range(values.shape[1]):
        for bucket in buckets.values():
            if len(bucket) > 1:
                source_order = rng.permutation(len(bucket))
                for target_index, source_index in enumerate(source_order):
                    permuted[bucket[target_index], column] = values[
                        bucket[source_index], column
                    ]
            else:
                indices = bucket[0]
                shift = int(rng.integers(0, len(indices))) if len(indices) > 1 else 0
                permuted[indices, column] = np.roll(values[indices, column], shift)
    return permuted


def _scaled_singular_values(matrix: FloatArray, center: bool) -> FloatArray:
    values = matrix - np.mean(matrix, axis=0, keepdims=True) if center else matrix
    denominator = np.sqrt(max(len(values) - int(center), 1))
    return np.linalg.svd(values, compute_uv=False) / denominator


def _leading_parallel_rank(observed: FloatArray, thresholds: FloatArray) -> int:
    result = 0
    for value, threshold in zip(observed, thresholds):
        if value <= threshold:
            break
        result += 1
    return result


@dataclass(frozen=True)
class RankSelectionResult:
    """Structured parallel-analysis rank estimate and block-bootstrap uncertainty."""

    selected_rank: int
    rank_ci: tuple[int, int]
    rank_probabilities: FloatArray
    observed_singular_values: FloatArray
    permutation_thresholds: FloatArray
    bootstrap_ranks: IntArray
    bootstrap_effective_ranks: FloatArray
    confidence: float
    n_bootstrap: int
    n_permutations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_rank": self.selected_rank,
            "rank_ci": self.rank_ci,
            "rank_probabilities": self.rank_probabilities.copy(),
            "observed_singular_values": self.observed_singular_values.copy(),
            "permutation_thresholds": self.permutation_thresholds.copy(),
            "bootstrap_ranks": self.bootstrap_ranks.copy(),
            "bootstrap_effective_ranks": self.bootstrap_effective_ranks.copy(),
            "confidence": self.confidence,
            "n_bootstrap": self.n_bootstrap,
            "n_permutations": self.n_permutations,
        }


def select_predictive_rank(
    signatures: ArrayLike,
    trajectory_ids: ArrayLike,
    *,
    state_group_ids: ArrayLike | None = None,
    n_bootstrap: int = 500,
    n_permutations: int = 500,
    confidence: float = 0.95,
    seed: int = 0,
    center: bool = True,
    within_state_groups: bool = True,
) -> RankSelectionResult:
    """Select finite-signature rank using structured parallel analysis.

    The permutation distribution supplies rank-specific noise thresholds.  The
    uncertainty interval then resamples independent state groups and complete
    trajectories while holding those thresholds fixed.  Selection is based on
    consecutive leading components, preventing a noisy late singular value
    from creating a disconnected rank estimate.
    """

    matrix = _finite_array(signatures, "signatures", ndim=2)
    trajectories, states = _validate_cluster_ids(
        len(matrix), trajectory_ids, state_group_ids
    )
    boot_count = int(n_bootstrap)
    permutation_count = int(n_permutations)
    level = float(confidence)
    if boot_count < 1 or permutation_count < 1:
        raise ValueError("n_bootstrap and n_permutations must be positive")
    if not 0 < level < 1:
        raise ValueError("confidence must lie strictly between zero and one")

    observed = _scaled_singular_values(matrix, bool(center))
    rng = np.random.default_rng(seed)
    null = np.empty((permutation_count, len(observed)), dtype=np.float64)
    for draw in range(permutation_count):
        permuted = structured_permute_columns(
            matrix,
            trajectories,
            state_group_ids=states,
            within_state_groups=within_state_groups,
            seed=int(rng.integers(0, np.iinfo(np.int64).max)),
        )
        null[draw] = _scaled_singular_values(permuted, bool(center))
    thresholds = np.quantile(null, level, axis=0)
    selected_rank = _leading_parallel_rank(observed, thresholds)

    top_units = _ordered_unique(trajectories if states is None else states)
    if len(top_units) < 2:
        raise ValueError("rank uncertainty requires at least two top-level clusters")
    ranks = np.empty(boot_count, dtype=np.int64)
    effective = np.empty(boot_count, dtype=np.float64)
    for draw in range(boot_count):
        indices = _one_hierarchical_bootstrap(trajectories, states, rng)
        singular = _scaled_singular_values(matrix[indices], bool(center))
        ranks[draw] = _leading_parallel_rank(singular, thresholds)
        effective[draw] = effective_rank_from_singular_values(singular)
    alpha = 1.0 - level
    low, high = np.quantile(
        ranks,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="nearest",
    )
    max_rank = len(observed)
    probabilities = np.bincount(ranks, minlength=max_rank + 1).astype(np.float64)
    probabilities /= boot_count
    return RankSelectionResult(
        selected_rank=int(selected_rank),
        rank_ci=(int(low), int(high)),
        rank_probabilities=probabilities,
        observed_singular_values=observed,
        permutation_thresholds=np.asarray(thresholds, dtype=np.float64),
        bootstrap_ranks=ranks,
        bootstrap_effective_ranks=effective,
        confidence=level,
        n_bootstrap=boot_count,
        n_permutations=permutation_count,
    )


# Explicit alias: this is parallel analysis, not universal Hankel-rank recovery.
structured_permutation_rank_selection = select_predictive_rank


# ---------------------------------------------------------------------------
# Affine-bilinear and hybrid dynamics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AffineBilinearOperator:
    """One mode of an affine-bilinear controlled state realization."""

    c: FloatArray
    A: FloatArray
    B: FloatArray
    N: FloatArray
    ridge: float = 0.0
    n_samples: int = 0
    training_rmse: float = np.nan

    def __post_init__(self) -> None:
        c = _finite_array(self.c, "c", ndim=1)
        A = _finite_array(self.A, "A", ndim=2)
        B = _finite_array(self.B, "B", ndim=2)
        N = _finite_array(self.N, "N", ndim=3)
        state_dim = len(c)
        if A.shape != (state_dim, state_dim):
            raise ValueError("A must be square with dimension len(c)")
        if B.shape[0] != state_dim or N.shape != (
            B.shape[1], state_dim, state_dim
        ):
            raise ValueError("B and N dimensions are inconsistent with A")
        if not np.isfinite(self.ridge) or self.ridge < 0:
            raise ValueError("ridge must be nonnegative and finite")
        if int(self.n_samples) < 0:
            raise ValueError("n_samples must be nonnegative")
        object.__setattr__(self, "c", c.copy())
        object.__setattr__(self, "A", A.copy())
        object.__setattr__(self, "B", B.copy())
        object.__setattr__(self, "N", N.copy())

    @property
    def state_dim(self) -> int:
        return int(self.A.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.B.shape[1])

    @property
    def intercept(self) -> FloatArray:
        return self.c


@dataclass(frozen=True)
class HybridAffineBilinearModel:
    """Mode-conditioned operators plus a separately fitted global baseline."""

    operators: Mapping[Hashable, AffineBilinearOperator]
    global_operator: AffineBilinearOperator

    def __post_init__(self) -> None:
        operators = dict(self.operators)
        if not operators:
            raise ValueError("operators must be nonempty")
        dimensions = {
            (operator.state_dim, operator.action_dim) for operator in operators.values()
        }
        dimensions.add(
            (self.global_operator.state_dim, self.global_operator.action_dim)
        )
        if len(dimensions) != 1:
            raise ValueError("all hybrid operators must have the same dimensions")
        object.__setattr__(self, "operators", operators)

    @property
    def state_dim(self) -> int:
        return self.global_operator.state_dim

    @property
    def action_dim(self) -> int:
        return self.global_operator.action_dim


def _state_action_training_arrays(
    states: ArrayLike, actions: ArrayLike, next_states: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    state = _finite_array(states, "states", ndim=2)
    action = _finite_array(actions, "actions", ndim=2)
    target = _finite_array(next_states, "next_states", ndim=2)
    if len(state) != len(action) or target.shape != state.shape:
        raise ValueError("states, actions, and next_states are not aligned")
    return state, action, target


def affine_bilinear_design(states: ArrayLike, actions: ArrayLike) -> FloatArray:
    """Construct ``[1, q, a, a_0 q, ..., a_{u-1} q]``."""

    state = _finite_array(states, "states", ndim=2)
    action = _finite_array(actions, "actions", ndim=2)
    if len(state) != len(action):
        raise ValueError("states and actions must have equal row counts")
    interactions = [state * action[:, index : index + 1] for index in range(action.shape[1])]
    return np.column_stack(
        [np.ones(len(state), dtype=np.float64), state, action, *interactions]
    )


def fit_affine_bilinear_dynamics(
    states: ArrayLike,
    actions: ArrayLike,
    next_states: ArrayLike,
    *,
    ridge: float = 1e-6,
) -> AffineBilinearOperator:
    """Fit one affine-bilinear operator with an unpenalized intercept."""

    state, action, target = _state_action_training_arrays(states, actions, next_states)
    penalty_value = float(ridge)
    if not np.isfinite(penalty_value) or penalty_value < 0:
        raise ValueError("ridge must be nonnegative and finite")
    design = affine_bilinear_design(state, action)
    if penalty_value == 0:
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    else:
        penalty = np.eye(design.shape[1], dtype=np.float64) * penalty_value
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ target,
        )
    state_dim = state.shape[1]
    action_dim = action.shape[1]
    action_start = 1 + state_dim
    interaction_start = action_start + action_dim
    c = coefficients[0]
    A = coefficients[1:action_start].T
    B = coefficients[action_start:interaction_start].T
    N = np.stack(
        [
            coefficients[
                interaction_start + index * state_dim : interaction_start
                + (index + 1) * state_dim
            ].T
            for index in range(action_dim)
        ],
        axis=0,
    )
    residual = target - design @ coefficients
    return AffineBilinearOperator(
        c=c,
        A=A,
        B=B,
        N=N,
        ridge=penalty_value,
        n_samples=len(state),
        training_rmse=float(np.sqrt(np.mean(residual**2))),
    )


# Common implementation-facing names.
fit_affine_bilinear_operator = fit_affine_bilinear_dynamics
fit_ridge_operator = fit_affine_bilinear_dynamics


def fit_mode_conditioned_dynamics(
    states: ArrayLike,
    actions: ArrayLike,
    next_states: ArrayLike,
    modes: ArrayLike,
    *,
    ridge: float = 1e-6,
    minimum_mode_samples: int = 1,
) -> HybridAffineBilinearModel:
    """Fit frozen-label hybrid operators and a capacity-comparable global baseline."""

    state, action, target = _state_action_training_arrays(states, actions, next_states)
    labels = np.asarray(modes).reshape(-1)
    if len(labels) != len(state):
        raise ValueError("modes must have one value per transition")
    minimum = int(minimum_mode_samples)
    if minimum < 1:
        raise ValueError("minimum_mode_samples must be positive")
    operators: dict[Hashable, AffineBilinearOperator] = {}
    for mode in _ordered_unique(labels):
        selected = labels == mode
        if int(np.sum(selected)) < minimum:
            raise ValueError(f"mode {mode!r} has fewer than minimum_mode_samples")
        operators[mode] = fit_affine_bilinear_dynamics(
            state[selected], action[selected], target[selected], ridge=ridge
        )
    global_operator = fit_affine_bilinear_dynamics(
        state, action, target, ridge=ridge
    )
    return HybridAffineBilinearModel(operators, global_operator)


fit_hybrid_affine_bilinear_dynamics = fit_mode_conditioned_dynamics


def _prediction_arrays(
    operator: AffineBilinearOperator,
    states: ArrayLike,
    actions: ArrayLike,
) -> tuple[FloatArray, FloatArray, bool]:
    state_raw = np.asarray(states, dtype=np.float64)
    action_raw = np.asarray(actions, dtype=np.float64)
    single = state_raw.ndim == 1
    if state_raw.ndim not in (1, 2):
        raise ValueError("states must be a state vector or a batch matrix")
    state = state_raw[None, :] if single else state_raw
    if state.shape[1] != operator.state_dim:
        raise ValueError("state dimension does not match the operator")
    if action_raw.ndim == 0:
        if operator.action_dim != 1:
            raise ValueError("a scalar action requires action_dim == 1")
        action = np.full((len(state), 1), float(action_raw))
    elif action_raw.ndim == 1:
        if single:
            action = action_raw[None, :]
        elif operator.action_dim == 1 and len(action_raw) == len(state):
            action = action_raw[:, None]
        elif len(action_raw) == operator.action_dim:
            action = np.broadcast_to(action_raw, (len(state), operator.action_dim))
        else:
            raise ValueError("one-dimensional actions are ambiguous or misaligned")
    elif action_raw.ndim == 2:
        action = action_raw
    else:
        raise ValueError("actions must be scalar, vector, or matrix")
    if action.shape != (len(state), operator.action_dim):
        raise ValueError("actions do not match the state batch and operator")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
        raise ValueError("states and actions must be finite")
    return np.asarray(state, dtype=np.float64), np.asarray(action, dtype=np.float64), single


def predict_affine_bilinear(
    operator: AffineBilinearOperator,
    states: ArrayLike,
    actions: ArrayLike,
) -> FloatArray:
    """Predict one step for one state or a batch."""

    state, action, single = _prediction_arrays(operator, states, actions)
    prediction = operator.c + state @ operator.A.T + action @ operator.B.T
    for index in range(operator.action_dim):
        prediction += action[:, index : index + 1] * (state @ operator.N[index].T)
    return prediction[0] if single else prediction


predict_operator = predict_affine_bilinear


def predict_mode_conditioned(
    model: HybridAffineBilinearModel,
    states: ArrayLike,
    actions: ArrayLike,
    modes: ArrayLike,
    *,
    unknown_mode: str = "error",
) -> FloatArray:
    """Predict with frozen modes; unknown labels fail closed unless ``global`` is requested."""

    state_raw = np.asarray(states, dtype=np.float64)
    single = state_raw.ndim == 1
    state = state_raw[None, :] if single else state_raw
    _, action, _ = _prediction_arrays(model.global_operator, state, actions)
    labels_raw = np.asarray(modes)
    if labels_raw.ndim == 0:
        labels = np.repeat(labels_raw.reshape(1), len(state))
    else:
        labels = labels_raw.reshape(-1)
    if len(labels) != len(state):
        raise ValueError("modes must have one value per state")
    if unknown_mode not in {"error", "global"}:
        raise ValueError("unknown_mode must be 'error' or 'global'")
    result = np.empty_like(state)
    for row, mode in enumerate(labels.tolist()):
        operator = model.operators.get(mode)
        if operator is None:
            if unknown_mode == "error":
                raise ValueError(f"unknown mode {mode!r}")
            operator = model.global_operator
        result[row] = predict_affine_bilinear(operator, state[row], action[row])
    return result[0] if single else result


predict_hybrid_dynamics = predict_mode_conditioned


def local_affine_map(
    operator: AffineBilinearOperator, action: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Return ``(M, d)`` such that a fixed action gives ``q_next = M q + d``."""

    value = _finite_array(action, "action").reshape(-1)
    if len(value) != operator.action_dim:
        raise ValueError("action dimension does not match the operator")
    matrix = operator.A + np.tensordot(value, operator.N, axes=(0, 0))
    offset = operator.c + operator.B @ value
    return np.asarray(matrix, dtype=np.float64), np.asarray(offset, dtype=np.float64)


def _action_sequence(actions: ArrayLike, action_dim: int) -> FloatArray:
    values = _finite_array(actions, "actions")
    if values.ndim == 1:
        values = values[:, None] if action_dim == 1 else values[None, :]
    if values.ndim != 2 or values.shape[1] != action_dim:
        raise ValueError("actions must be a horizon-by-action_dim matrix")
    return values


def _operator_for_step(
    model: AffineBilinearOperator | HybridAffineBilinearModel,
    mode: Any | None,
) -> AffineBilinearOperator:
    if isinstance(model, AffineBilinearOperator):
        return model
    if mode not in model.operators:
        raise ValueError(f"unknown or missing mode {mode!r}")
    return model.operators[mode]


def compose_affine_bilinear_dynamics(
    model: AffineBilinearOperator | HybridAffineBilinearModel,
    actions: ArrayLike,
    *,
    modes: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Compose a fixed action/mode word into ``q_H = M q_0 + d``."""

    action_dim = model.action_dim
    sequence = _action_sequence(actions, action_dim)
    if isinstance(model, HybridAffineBilinearModel):
        if modes is None:
            raise ValueError("hybrid composition requires one frozen mode per action")
        labels = np.asarray(modes).reshape(-1)
        if len(labels) != len(sequence):
            raise ValueError("modes and actions must have equal horizons")
    else:
        labels = np.asarray([None] * len(sequence), dtype=object)
    total_matrix = np.eye(model.state_dim, dtype=np.float64)
    total_offset = np.zeros(model.state_dim, dtype=np.float64)
    for action, mode in zip(sequence, labels.tolist()):
        operator = _operator_for_step(model, mode)
        step_matrix, step_offset = local_affine_map(operator, action)
        total_offset = step_matrix @ total_offset + step_offset
        total_matrix = step_matrix @ total_matrix
    return total_matrix, total_offset


compose_operators = compose_affine_bilinear_dynamics


def rollout_affine_bilinear(
    model: AffineBilinearOperator | HybridAffineBilinearModel,
    initial_state: ArrayLike,
    actions: ArrayLike,
    *,
    modes: ArrayLike | None = None,
) -> FloatArray:
    """Roll out a fixed action word, returning the initial state and every successor."""

    initial = _finite_array(initial_state, "initial_state", ndim=1)
    if len(initial) != model.state_dim:
        raise ValueError("initial_state dimension does not match the model")
    sequence = _action_sequence(actions, model.action_dim)
    if isinstance(model, HybridAffineBilinearModel):
        if modes is None:
            raise ValueError("hybrid rollout requires one frozen mode per action")
        labels = np.asarray(modes).reshape(-1)
        if len(labels) != len(sequence):
            raise ValueError("modes and actions must have equal horizons")
    else:
        labels = np.asarray([None] * len(sequence), dtype=object)
    trajectory = np.empty((len(sequence) + 1, model.state_dim), dtype=np.float64)
    trajectory[0] = initial
    for step, (action, mode) in enumerate(zip(sequence, labels.tolist())):
        operator = _operator_for_step(model, mode)
        trajectory[step + 1] = predict_affine_bilinear(
            operator, trajectory[step], action
        )
    return trajectory


rollout_operator = rollout_affine_bilinear


# ---------------------------------------------------------------------------
# Fixed whitened transport, conjugacy, and intertwining
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WhitenedProcrustesMap:
    """One calibration-fitted invertible affine map between equal-dimensional states."""

    matrix: FloatArray
    offset: FloatArray
    source_mean: FloatArray
    target_mean: FloatArray
    condition_number: float
    source_covariance_condition: float
    target_covariance_condition: float
    calibration_relative_rmse: float

    def __post_init__(self) -> None:
        matrix = _finite_array(self.matrix, "matrix", ndim=2)
        offset = _finite_array(self.offset, "offset", ndim=1)
        if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(offset):
            raise ValueError("transport matrix must be square and match the offset")
        if np.linalg.matrix_rank(matrix) != len(offset):
            raise ValueError("transport matrix must be invertible")
        object.__setattr__(self, "matrix", matrix.copy())
        object.__setattr__(self, "offset", offset.copy())
        object.__setattr__(
            self, "source_mean", _finite_array(self.source_mean, "source_mean", ndim=1).copy()
        )
        object.__setattr__(
            self, "target_mean", _finite_array(self.target_mean, "target_mean", ndim=1).copy()
        )

    def apply(self, states: ArrayLike) -> FloatArray:
        return apply_fixed_map(self, states)

    def inverse_apply(self, states: ArrayLike) -> FloatArray:
        values = np.asarray(states, dtype=np.float64)
        if values.shape[-1] != len(self.offset) or not np.all(np.isfinite(values)):
            raise ValueError("states do not match the transport map")
        inverse = np.linalg.inv(self.matrix)
        return (values - self.offset) @ inverse.T


FixedLinearMap = WhitenedProcrustesMap


def _symmetric_root(matrix: FloatArray, inverse: bool) -> tuple[FloatArray, FloatArray]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if np.any(eigenvalues <= 0):
        raise ValueError("covariance is not positive definite")
    powers = 1.0 / np.sqrt(eigenvalues) if inverse else np.sqrt(eigenvalues)
    return (eigenvectors * powers) @ eigenvectors.T, eigenvalues


def fit_whitened_procrustes(
    source: ArrayLike,
    target: ArrayLike,
    *,
    regularization: float = 1e-10,
    max_condition: float = 1e8,
) -> WhitenedProcrustesMap:
    """Fit one affine, invertible, whitened-Procrustes map on calibration rows.

    Both covariance charts and the recovered map must satisfy ``max_condition``.
    The function intentionally supports equal ranks only; a rectangular map is
    not a similarity transformation.
    """

    first = _finite_array(source, "source", ndim=2)
    second = _finite_array(target, "target", ndim=2)
    if first.shape != second.shape:
        raise ValueError("source and target must have the same calibration shape")
    if first.shape[0] < 2:
        raise ValueError("at least two calibration rows are required")
    ridge = float(regularization)
    limit = float(max_condition)
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("regularization must be nonnegative and finite")
    if not np.isfinite(limit) or limit <= 1:
        raise ValueError("max_condition must be finite and greater than one")
    source_mean = np.mean(first, axis=0)
    target_mean = np.mean(second, axis=0)
    centered_source = first - source_mean
    centered_target = second - target_mean
    divisor = max(len(first) - 1, 1)
    source_covariance = centered_source.T @ centered_source / divisor
    target_covariance = centered_target.T @ centered_target / divisor
    source_covariance += ridge * np.eye(first.shape[1])
    target_covariance += ridge * np.eye(first.shape[1])
    source_inverse_root, source_eigenvalues = _symmetric_root(
        source_covariance, True
    )
    target_inverse_root, target_eigenvalues = _symmetric_root(
        target_covariance, True
    )
    source_condition = float(source_eigenvalues[-1] / source_eigenvalues[0])
    target_condition = float(target_eigenvalues[-1] / target_eigenvalues[0])
    if source_condition > limit or target_condition > limit:
        raise ValueError("calibration covariance exceeds max_condition")
    target_root, _ = _symmetric_root(target_covariance, False)
    whitened_source = centered_source @ source_inverse_root
    whitened_target = centered_target @ target_inverse_root
    left, _, right_t = np.linalg.svd(
        whitened_source.T @ whitened_target, full_matrices=False
    )
    orthogonal_row_map = left @ right_t
    row_map = source_inverse_root @ orthogonal_row_map @ target_root
    matrix = row_map.T
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > limit:
        raise ValueError("recovered map is singular or exceeds max_condition")
    offset = target_mean - matrix @ source_mean
    fitted = first @ matrix.T + offset
    residual_rms = float(np.sqrt(np.mean((fitted - second) ** 2)))
    target_rms = float(np.sqrt(np.mean(centered_target**2)))
    relative = residual_rms / max(target_rms, np.finfo(np.float64).tiny)
    return WhitenedProcrustesMap(
        matrix=matrix,
        offset=offset,
        source_mean=source_mean,
        target_mean=target_mean,
        condition_number=condition,
        source_covariance_condition=source_condition,
        target_covariance_condition=target_condition,
        calibration_relative_rmse=float(relative),
    )


fit_fixed_procrustes_map = fit_whitened_procrustes


def apply_fixed_map(
    transport: WhitenedProcrustesMap | ArrayLike,
    states: ArrayLike,
) -> FloatArray:
    """Apply a fixed map to a state vector or row batch."""

    values = np.asarray(states, dtype=np.float64)
    if isinstance(transport, WhitenedProcrustesMap):
        matrix = transport.matrix
        offset = transport.offset
    else:
        matrix = _finite_array(transport, "transport", ndim=2)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("transport matrix must be square")
        offset = np.zeros(matrix.shape[0], dtype=np.float64)
    if values.ndim not in (1, 2) or values.shape[-1] != matrix.shape[1]:
        raise ValueError("states do not match the transport map")
    if not np.all(np.isfinite(values)):
        raise ValueError("states contain nonfinite values")
    return values @ matrix.T + offset


def _transport_parts(
    transport: WhitenedProcrustesMap | ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    if isinstance(transport, WhitenedProcrustesMap):
        return transport.matrix, transport.offset
    matrix = _finite_array(transport, "transport", ndim=2)
    if matrix.shape[0] != matrix.shape[1] or np.linalg.matrix_rank(matrix) != len(matrix):
        raise ValueError("transport must be square and invertible")
    return matrix, np.zeros(len(matrix), dtype=np.float64)


def conjugate_affine_bilinear_operator(
    source: AffineBilinearOperator,
    transport: WhitenedProcrustesMap | ArrayLike,
) -> AffineBilinearOperator:
    """Express an operator under ``q_target = S q_source + offset``."""

    matrix, offset = _transport_parts(transport)
    if matrix.shape != (source.state_dim, source.state_dim):
        raise ValueError("transport and source dimensions differ")
    inverse = np.linalg.inv(matrix)
    A = matrix @ source.A @ inverse
    N = np.stack([matrix @ item @ inverse for item in source.N], axis=0)
    B = matrix @ source.B
    for index in range(source.action_dim):
        B[:, index] -= N[index] @ offset
    c = matrix @ source.c + offset - A @ offset
    return AffineBilinearOperator(c=c, A=A, B=B, N=N)


def _relative_error(actual: FloatArray, expected: FloatArray) -> float:
    numerator = float(np.linalg.norm(actual - expected))
    denominator = max(
        float(np.linalg.norm(actual)),
        float(np.linalg.norm(expected)),
        np.finfo(np.float64).tiny,
    )
    return numerator / denominator


def operator_conjugacy_metrics(
    source: AffineBilinearOperator,
    target: AffineBilinearOperator,
    transport: WhitenedProcrustesMap | ArrayLike,
) -> dict[str, Any]:
    """Relative errors after explicitly conjugating every affine-bilinear block."""

    if (source.state_dim, source.action_dim) != (target.state_dim, target.action_dim):
        raise ValueError("source and target operator dimensions differ")
    expected = conjugate_affine_bilinear_operator(source, transport)
    residual_parts = [
        target.c - expected.c,
        target.A - expected.A,
        target.B - expected.B,
        target.N - expected.N,
    ]
    reference_parts = [target.c, target.A, target.B, target.N]
    numerator = np.sqrt(sum(float(np.sum(part**2)) for part in residual_parts))
    denominator = np.sqrt(sum(float(np.sum(part**2)) for part in reference_parts))
    aggregate = numerator / max(denominator, np.finfo(np.float64).tiny)
    return {
        "aggregate_relative_error": float(aggregate),
        "intercept_relative_error": _relative_error(target.c, expected.c),
        "A_relative_error": _relative_error(target.A, expected.A),
        "B_relative_error": _relative_error(target.B, expected.B),
        "N_relative_error": _relative_error(target.N, expected.N),
        "expected_operator": expected,
    }


def operator_intertwining_metrics(
    source: AffineBilinearOperator,
    target: AffineBilinearOperator,
    transport: WhitenedProcrustesMap | ArrayLike,
) -> dict[str, Any]:
    """Check intertwining equations directly, without numerically using ``S^-1``."""

    matrix, offset = _transport_parts(transport)
    if (source.state_dim, source.action_dim) != (target.state_dim, target.action_dim):
        raise ValueError("source and target operator dimensions differ")
    if matrix.shape != (source.state_dim, source.state_dim):
        raise ValueError("transport dimension differs from the operators")
    A_left, A_right = target.A @ matrix, matrix @ source.A
    N_left = np.einsum("uij,jk->uik", target.N, matrix)
    N_right = np.einsum("ij,ujk->uik", matrix, source.N)
    B_left = target.B.copy()
    for index in range(target.action_dim):
        B_left[:, index] += target.N[index] @ offset
    B_right = matrix @ source.B
    c_left = target.c + target.A @ offset
    c_right = matrix @ source.c + offset
    residual_parts = [A_left - A_right, N_left - N_right, B_left - B_right, c_left - c_right]
    reference_parts = [A_left, N_left, B_left, c_left, A_right, N_right, B_right, c_right]
    numerator = np.sqrt(sum(float(np.sum(part**2)) for part in residual_parts))
    denominator = np.sqrt(
        0.5 * sum(float(np.sum(part**2)) for part in reference_parts)
    )
    return {
        "aggregate_relative_error": float(
            numerator / max(denominator, np.finfo(np.float64).tiny)
        ),
        "intercept_relative_error": _relative_error(c_left, c_right),
        "A_relative_error": _relative_error(A_left, A_right),
        "B_relative_error": _relative_error(B_left, B_right),
        "N_relative_error": _relative_error(N_left, N_right),
    }


def hybrid_operator_conjugacy_metrics(
    source: HybridAffineBilinearModel | Mapping[Hashable, AffineBilinearOperator],
    target: HybridAffineBilinearModel | Mapping[Hashable, AffineBilinearOperator],
    transport: WhitenedProcrustesMap | ArrayLike,
) -> dict[str, Any]:
    """Require one fixed map to conjugate every named mode."""

    source_operators = source.operators if isinstance(source, HybridAffineBilinearModel) else dict(source)
    target_operators = target.operators if isinstance(target, HybridAffineBilinearModel) else dict(target)
    if set(source_operators) != set(target_operators):
        raise ValueError("source and target have different mode labels")
    by_mode = {
        mode: operator_conjugacy_metrics(
            source_operators[mode], target_operators[mode], transport
        )
        for mode in source_operators
    }
    errors = np.asarray(
        [metrics["aggregate_relative_error"] for metrics in by_mode.values()],
        dtype=np.float64,
    )
    return {
        "aggregate_relative_error": float(np.sqrt(np.mean(errors**2))),
        "maximum_mode_relative_error": float(np.max(errors)),
        "by_mode": by_mode,
        "mode_count": len(by_mode),
    }


hybrid_conjugacy_metrics = hybrid_operator_conjugacy_metrics


# ---------------------------------------------------------------------------
# Reachability, observability, and internal interchange
# ---------------------------------------------------------------------------


def _matrix_diagnostics(matrix: FloatArray, tolerance: float | None) -> dict[str, Any]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size == 0 or singular[0] == 0:
        rank = 0
        condition = np.inf
    else:
        threshold = (
            max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
            if tolerance is None
            else float(tolerance)
        )
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("tolerance must be nonnegative and finite")
        rank = int(np.sum(singular > threshold))
        condition = (
            float(singular[0] / singular[rank - 1]) if rank == min(matrix.shape) else np.inf
        )
    return {
        "matrix": matrix,
        "singular_values": singular,
        "rank": rank,
        "full_rank": bool(rank == min(matrix.shape)),
        "condition_number": condition,
        "entropy_effective_rank": effective_rank_from_singular_values(singular),
        "stable_rank": effective_rank_from_singular_values(singular, method="stable"),
    }


def reachability_diagnostics(
    operator: AffineBilinearOperator,
    *,
    reference_state: ArrayLike | None = None,
    reference_action: ArrayLike | None = None,
    horizon: int | None = None,
    tolerance: float | None = None,
) -> dict[str, Any]:
    """Local reachability of the bilinear system around a reference state/action."""

    state = (
        np.zeros(operator.state_dim, dtype=np.float64)
        if reference_state is None
        else _finite_array(reference_state, "reference_state", ndim=1)
    )
    action = (
        np.zeros(operator.action_dim, dtype=np.float64)
        if reference_action is None
        else _finite_array(reference_action, "reference_action", ndim=1)
    )
    if len(state) != operator.state_dim or len(action) != operator.action_dim:
        raise ValueError("reference state/action dimensions do not match the operator")
    steps = operator.state_dim if horizon is None else int(horizon)
    if steps < 1:
        raise ValueError("horizon must be positive")
    transition, _ = local_affine_map(operator, action)
    input_jacobian = operator.B.copy()
    for index in range(operator.action_dim):
        input_jacobian[:, index] += operator.N[index] @ state
    blocks: list[FloatArray] = []
    power = np.eye(operator.state_dim, dtype=np.float64)
    for _ in range(steps):
        blocks.append(power @ input_jacobian)
        power = transition @ power
    result = _matrix_diagnostics(np.column_stack(blocks), tolerance)
    result.update({"transition_jacobian": transition, "input_jacobian": input_jacobian})
    return result


def observability_diagnostics(
    operator: AffineBilinearOperator,
    observation_matrix: ArrayLike,
    *,
    reference_action: ArrayLike | None = None,
    horizon: int | None = None,
    tolerance: float | None = None,
) -> dict[str, Any]:
    """Local observability for a specified physical readout matrix ``C``."""

    observation = _finite_array(observation_matrix, "observation_matrix", ndim=2)
    if observation.shape[1] != operator.state_dim:
        raise ValueError("observation_matrix has the wrong state dimension")
    action = (
        np.zeros(operator.action_dim, dtype=np.float64)
        if reference_action is None
        else _finite_array(reference_action, "reference_action", ndim=1)
    )
    if len(action) != operator.action_dim:
        raise ValueError("reference_action has the wrong dimension")
    steps = operator.state_dim if horizon is None else int(horizon)
    if steps < 1:
        raise ValueError("horizon must be positive")
    transition, _ = local_affine_map(operator, action)
    blocks: list[FloatArray] = []
    power = np.eye(operator.state_dim, dtype=np.float64)
    for _ in range(steps):
        blocks.append(observation @ power)
        power = power @ transition
    result = _matrix_diagnostics(np.vstack(blocks), tolerance)
    result.update({"transition_jacobian": transition, "observation_matrix": observation})
    return result


def hybrid_reachability_diagnostics(
    model: HybridAffineBilinearModel,
    initial_state: ArrayLike,
    actions: ArrayLike,
    modes: ArrayLike,
    *,
    tolerance: float | None = None,
) -> dict[str, Any]:
    """Endpoint input-Jacobian rank for one frozen hybrid action/mode word."""

    sequence = _action_sequence(actions, model.action_dim)
    labels = np.asarray(modes).reshape(-1)
    if len(labels) != len(sequence):
        raise ValueError("modes and actions must have equal horizons")
    trajectory = rollout_affine_bilinear(
        model, initial_state, sequence, modes=labels
    )
    transitions: list[FloatArray] = []
    inputs: list[FloatArray] = []
    for step, (action, mode) in enumerate(zip(sequence, labels.tolist())):
        operator = _operator_for_step(model, mode)
        transition, _ = local_affine_map(operator, action)
        input_jacobian = operator.B.copy()
        for index in range(operator.action_dim):
            input_jacobian[:, index] += operator.N[index] @ trajectory[step]
        transitions.append(transition)
        inputs.append(input_jacobian)
    endpoint_blocks: list[FloatArray] = []
    for start, input_jacobian in enumerate(inputs):
        propagated = input_jacobian
        for step in range(start + 1, len(transitions)):
            propagated = transitions[step] @ propagated
        endpoint_blocks.append(propagated)
    result = _matrix_diagnostics(np.column_stack(endpoint_blocks), tolerance)
    result.update({"trajectory": trajectory, "transition_jacobians": tuple(transitions)})
    return result


def internal_interchange_metrics(
    baseline: ArrayLike,
    interchanged: ArrayLike,
    intended_counterfactual: ArrayLike,
    *,
    minimum_effect_energy: float = 1e-12,
) -> dict[str, Any]:
    """Score whether an actual internal interchange realizes its intended effect.

    This is deliberately evaluated in held-out physical/predictive coordinates,
    not by asking whether an externally decoded state can be linearly aligned.
    """

    base = _finite_array(baseline, "baseline")
    observed = _finite_array(interchanged, "interchanged")
    intended = _finite_array(intended_counterfactual, "intended_counterfactual")
    if base.shape != observed.shape or base.shape != intended.shape:
        raise ValueError("baseline, interchanged, and intended arrays must match")
    floor = float(minimum_effect_energy)
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("minimum_effect_energy must be positive and finite")
    observed_effect = observed - base
    intended_effect = intended - base
    intended_energy = float(np.sum(intended_effect**2))
    observed_energy = float(np.sum(observed_effect**2))
    if intended_energy < floor:
        return {
            "eligible": False,
            "intended_effect_energy": intended_energy,
            "observed_effect_energy": observed_energy,
            "effect_cosine": np.nan,
            "relative_effect_error": np.nan,
            "bounded_effect_error": np.nan,
            "counterfactual_rmse": np.nan,
            "improvement_over_baseline": np.nan,
        }
    error = float(np.linalg.norm(observed_effect - intended_effect))
    intended_norm = float(np.sqrt(intended_energy))
    observed_norm = float(np.sqrt(observed_energy))
    denominator = intended_norm * observed_norm
    cosine = (
        float(np.sum(observed_effect * intended_effect) / denominator)
        if denominator > np.finfo(np.float64).tiny
        else 0.0
    )
    counterfactual_mse = float(np.mean((observed - intended) ** 2))
    baseline_mse = float(np.mean((base - intended) ** 2))
    relative_error = error / intended_norm
    return {
        "eligible": True,
        "intended_effect_energy": intended_energy,
        "observed_effect_energy": observed_energy,
        "effect_cosine": float(np.clip(cosine, -1.0, 1.0)),
        "relative_effect_error": float(relative_error),
        "bounded_effect_error": float(relative_error / (1.0 + relative_error)),
        "counterfactual_rmse": float(np.sqrt(counterfactual_mse)),
        "improvement_over_baseline": float(
            1.0 - counterfactual_mse / max(baseline_mse, np.finfo(np.float64).tiny)
        ),
    }


def transported_internal_interchange_metrics(
    source_baseline: ArrayLike,
    source_intervened: ArrayLike,
    target_baseline: ArrayLike,
    target_interchanged: ArrayLike,
    intended_target_counterfactual: ArrayLike,
    transport: WhitenedProcrustesMap | ArrayLike,
    *,
    minimum_effect_energy: float = 1e-12,
) -> dict[str, Any]:
    """Combine source-effect transport and actual target-interchange diagnostics."""

    source_base = _finite_array(source_baseline, "source_baseline")
    source_edit = _finite_array(source_intervened, "source_intervened")
    target_base = _finite_array(target_baseline, "target_baseline")
    if source_base.shape != source_edit.shape or source_base.shape != target_base.shape:
        raise ValueError("source and target baseline batches must have matching shapes")
    matrix, _ = _transport_parts(transport)
    transported_effect = (source_edit - source_base) @ matrix.T
    intended = _finite_array(
        intended_target_counterfactual, "intended_target_counterfactual"
    )
    if intended.shape != target_base.shape:
        raise ValueError("intended target counterfactual has the wrong shape")
    predicted_from_source = target_base + transported_effect
    actual = internal_interchange_metrics(
        target_base,
        target_interchanged,
        intended,
        minimum_effect_energy=minimum_effect_energy,
    )
    transported = internal_interchange_metrics(
        target_base,
        predicted_from_source,
        intended,
        minimum_effect_energy=minimum_effect_energy,
    )
    return {
        **{f"actual_{key}": value for key, value in actual.items()},
        **{f"transported_{key}": value for key, value in transported.items()},
        "transported_prediction": predicted_from_source,
    }


causal_interchange_metrics = internal_interchange_metrics


# ---------------------------------------------------------------------------
# Clustered uncertainty, multiplicity, planning, and decision gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusteredBootstrapCI:
    estimate: float
    low: float
    high: float
    confidence: float
    draws: FloatArray
    n_top_level_clusters: int
    n_bootstrap: int

    def as_dict(self, *, include_draws: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "estimate": self.estimate,
            "low": self.low,
            "high": self.high,
            "confidence": self.confidence,
            "n_top_level_clusters": self.n_top_level_clusters,
            "n_bootstrap": self.n_bootstrap,
        }
        if include_draws:
            result["draws"] = self.draws.copy()
        return result


def clustered_bootstrap_ci(
    values: ArrayLike,
    trajectory_ids: ArrayLike,
    *,
    state_group_ids: ArrayLike | None = None,
    statistic: Callable[[FloatArray], float] | None = None,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> ClusteredBootstrapCI:
    """Percentile CI resampling whole trajectories and optional state groups."""

    array = _finite_array(values, "values")
    if array.ndim == 0:
        raise ValueError("values must have a sample axis")
    trajectories, states = _validate_cluster_ids(
        len(array), trajectory_ids, state_group_ids
    )
    level = float(confidence)
    repetitions = int(n_bootstrap)
    if not 0 < level < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if repetitions < 1:
        raise ValueError("n_bootstrap must be positive")
    top_units = _ordered_unique(trajectories if states is None else states)
    if len(top_units) < 2:
        raise ValueError("at least two independent top-level clusters are required")
    function = statistic if statistic is not None else lambda sample: float(np.mean(sample))
    estimate = float(function(array))
    if not np.isfinite(estimate):
        raise ValueError("statistic returned a nonfinite estimate")
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for draw in range(repetitions):
        indices = _one_hierarchical_bootstrap(trajectories, states, rng)
        draws[draw] = float(function(array[indices]))
    if not np.all(np.isfinite(draws)):
        raise ValueError("statistic returned a nonfinite bootstrap draw")
    alpha = 1.0 - level
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return ClusteredBootstrapCI(
        estimate=estimate,
        low=float(low),
        high=float(high),
        confidence=level,
        draws=draws,
        n_top_level_clusters=len(top_units),
        n_bootstrap=repetitions,
    )


clustered_bootstrap_interval = clustered_bootstrap_ci


def holm_correction(p_values: ArrayLike, *, alpha: float = 0.05) -> dict[str, Any]:
    """Holm step-down familywise-error correction, preserving input shape."""

    values = np.asarray(p_values, dtype=np.float64)
    original_shape = values.shape
    flat = values.reshape(-1)
    level = float(alpha)
    if flat.size == 0 or not np.all(np.isfinite(flat)):
        raise ValueError("p_values must be finite and nonempty")
    if np.any((flat < 0) | (flat > 1)):
        raise ValueError("p_values must lie in [0, 1]")
    if not 0 < level < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    order = np.argsort(flat, kind="stable")
    sorted_values = flat[order]
    factors = np.arange(len(flat), 0, -1, dtype=np.float64)
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(factors * sorted_values))
    adjusted = np.empty_like(flat)
    adjusted[order] = adjusted_sorted
    reject = adjusted <= level
    return {
        "adjusted_p_values": adjusted.reshape(original_shape),
        "reject": reject.reshape(original_shape),
        "order": order,
        "alpha": level,
    }


holm_bonferroni = holm_correction


def planning_decision_metrics(
    predicted_costs: ArrayLike,
    oracle_costs: ArrayLike,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Evaluate exhaustive action selection against held-out physical costs."""

    predicted = _finite_array(predicted_costs, "predicted_costs")
    oracle = _finite_array(oracle_costs, "oracle_costs")
    if predicted.shape != oracle.shape or predicted.ndim < 1:
        raise ValueError("predicted_costs and oracle_costs must have the same shape")
    if predicted.shape[-1] < 2:
        raise ValueError("planning requires at least two candidate actions")
    tie = float(tie_tolerance)
    if not np.isfinite(tie) or tie < 0:
        raise ValueError("tie_tolerance must be nonnegative and finite")
    action_count = predicted.shape[-1]
    predicted_rows = predicted.reshape(-1, action_count)
    oracle_rows = oracle.reshape(-1, action_count)
    selected = np.argmin(predicted_rows, axis=1)
    minimum = np.min(oracle_rows, axis=1)
    maximum = np.max(oracle_rows, axis=1)
    selected_cost = oracle_rows[np.arange(len(oracle_rows)), selected]
    regret = selected_cost - minimum
    spread = maximum - minimum
    normalized = np.divide(
        regret,
        spread,
        out=np.zeros_like(regret),
        where=spread > tie,
    )
    top1 = selected_cost <= minimum + tie
    return {
        "selected_actions": selected.reshape(predicted.shape[:-1]),
        "regret": regret.reshape(predicted.shape[:-1]),
        "normalized_regret": normalized.reshape(predicted.shape[:-1]),
        "top1_correct": top1.reshape(predicted.shape[:-1]),
        "top1_accuracy": float(np.mean(top1)),
        "mean_regret": float(np.mean(regret)),
        "mean_normalized_regret": float(np.mean(normalized)),
        "n_decisions": len(oracle_rows),
    }


def planning_gate(
    metrics: Mapping[str, Any],
    *,
    minimum_top1_accuracy: float = 0.5,
    maximum_mean_normalized_regret: float = 0.25,
) -> dict[str, Any]:
    """Apply prespecified planning thresholds to ``planning_decision_metrics``."""

    accuracy_threshold = float(minimum_top1_accuracy)
    regret_threshold = float(maximum_mean_normalized_regret)
    if not 0 <= accuracy_threshold <= 1 or not 0 <= regret_threshold <= 1:
        raise ValueError("planning thresholds must lie in [0, 1]")
    accuracy = float(metrics["top1_accuracy"])
    regret = float(metrics["mean_normalized_regret"])
    checks = {
        "top1_accuracy": bool(accuracy >= accuracy_threshold),
        "normalized_regret": bool(regret <= regret_threshold),
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def stage33_decision_gate(
    *,
    stable_low_rank: bool,
    hybrid_improvement: bool,
    fixed_map_well_conditioned: bool,
    heldout_conjugacy: bool,
    internal_interchange: bool,
    planning_value: bool,
    controls_rejected: bool,
    family_consistency: bool = True,
) -> dict[str, Any]:
    """Cumulative preregistered gate for a shared-mechanism conclusion.

    ``pass`` requires every condition.  ``partial_pass`` requires at least a
    stable realization, hybrid benefit, one valid fixed map, held-out operator
    conjugacy, negative controls, and family consistency, but may lack causal
    interchange and/or planning value.  Anything weaker is ``fail`` and cannot
    support a shared-mechanism statement.
    """

    checks = {
        "stable_low_rank": bool(stable_low_rank),
        "hybrid_improvement": bool(hybrid_improvement),
        "fixed_map_well_conditioned": bool(fixed_map_well_conditioned),
        "heldout_conjugacy": bool(heldout_conjugacy),
        "internal_interchange": bool(internal_interchange),
        "planning_value": bool(planning_value),
        "controls_rejected": bool(controls_rejected),
        "family_consistency": bool(family_consistency),
    }
    passed = all(checks.values())
    partial_requirements = (
        checks["stable_low_rank"]
        and checks["hybrid_improvement"]
        and checks["fixed_map_well_conditioned"]
        and checks["heldout_conjugacy"]
        and checks["controls_rejected"]
        and checks["family_consistency"]
    )
    if passed:
        status = "pass"
        evidence_level = 6
    elif partial_requirements:
        status = "partial_pass"
        evidence_level = 5 if checks["internal_interchange"] else 4
    else:
        status = "fail"
        evidence_level = 3 if checks["stable_low_rank"] else 1
    return {
        "status": status,
        "passed": passed,
        "evidence_level": evidence_level,
        "checks": checks,
        "failed_checks": tuple(name for name, value in checks.items() if not value),
    }


decision_gate = stage33_decision_gate


__all__ = [
    "AffineBilinearOperator",
    "ClusteredBootstrapCI",
    "FixedLinearMap",
    "HybridAffineBilinearModel",
    "RankSelectionResult",
    "WhitenedProcrustesMap",
    "affine_bilinear_design",
    "apply_fixed_map",
    "block_bootstrap_indices",
    "bounded_predictive_signature_pseudometric",
    "causal_interchange_metrics",
    "clustered_bootstrap_ci",
    "clustered_bootstrap_interval",
    "compose_affine_bilinear_dynamics",
    "compose_operators",
    "conjugate_affine_bilinear_operator",
    "decision_gate",
    "effective_rank",
    "effective_rank_from_singular_values",
    "epsilon_predictively_related",
    "fit_affine_bilinear_dynamics",
    "fit_affine_bilinear_operator",
    "fit_fixed_procrustes_map",
    "fit_hybrid_affine_bilinear_dynamics",
    "fit_mode_conditioned_dynamics",
    "fit_ridge_operator",
    "fit_whitened_procrustes",
    "holm_bonferroni",
    "holm_correction",
    "hybrid_conjugacy_metrics",
    "hybrid_operator_conjugacy_metrics",
    "hybrid_reachability_diagnostics",
    "internal_interchange_metrics",
    "local_affine_map",
    "numerical_rank",
    "observability_diagnostics",
    "operator_conjugacy_metrics",
    "operator_intertwining_metrics",
    "pairwise_predictive_signature_distances",
    "planning_decision_metrics",
    "planning_gate",
    "predict_affine_bilinear",
    "predict_hybrid_dynamics",
    "predict_mode_conditioned",
    "predict_operator",
    "predictive_signature_distance",
    "reachability_diagnostics",
    "rollout_affine_bilinear",
    "rollout_operator",
    "select_predictive_rank",
    "stage33_decision_gate",
    "structured_permutation_rank_selection",
    "structured_permute_columns",
    "transported_internal_interchange_metrics",
]


# ---------------------------------------------------------------------------
# Self-contained notebook API
# ---------------------------------------------------------------------------
#
# The Stage 33 notebook builder copies only the source text of the functions
# named below.  Consequently these definitions intentionally use only NumPy
# and built-ins (no private module helpers or dataclass annotations).  They
# also accept the dictionary representation saved in result bundles.  The
# richer typed functions above remain the repository-facing implementation.


def signature_pseudometric(
    left,
    right,
    scales=None,
    weights=None,
    bound=1.0,
):
    """Bounded weighted-L2 pseudometric on a frozen predictive signature bank."""

    first = np.asarray(left, dtype=np.float64).reshape(-1)
    second = np.asarray(right, dtype=np.float64).reshape(-1)
    if first.size == 0 or first.shape != second.shape:
        raise ValueError("signatures must have the same nonempty shape")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("signatures must be finite")
    limit = float(bound)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("bound must be positive and finite")
    scale = (
        np.ones_like(first)
        if scales is None
        else np.asarray(scales, dtype=np.float64).reshape(-1)
    )
    weight = (
        np.ones_like(first)
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(-1)
    )
    if (
        scale.shape != first.shape
        or weight.shape != first.shape
        or not np.all(np.isfinite(scale))
        or not np.all(np.isfinite(weight))
        or np.any(scale <= 0)
        or np.any(weight < 0)
        or not np.any(weight > 0)
    ):
        raise ValueError("scales/weights are invalid or do not match the signatures")
    distance = np.sqrt(
        np.sum(weight * ((first - second) / scale) ** 2) / np.sum(weight)
    )
    return float(min(limit, distance))


def effective_rank(values, method="entropy", center=False):  # noqa: F811
    """Entropy effective rank (or stable rank) of a matrix or SVD spectrum."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("values must be finite and nonempty")
    if array.ndim == 1:
        singular = array.reshape(-1)
        if np.any(singular < 0):
            raise ValueError("a supplied singular-value spectrum must be nonnegative")
    elif array.ndim == 2:
        matrix = array - np.mean(array, axis=0, keepdims=True) if center else array
        singular = np.linalg.svd(matrix, compute_uv=False)
    else:
        raise ValueError("values must be a matrix or a singular-value vector")
    energy = singular**2
    total = float(np.sum(energy))
    if total <= np.finfo(np.float64).tiny:
        return 0.0
    if method == "stable":
        return float(total / np.max(energy))
    if method != "entropy":
        raise ValueError("method must be 'entropy' or 'stable'")
    probabilities = energy[energy > 0] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def select_stable_rank(
    matrix,
    groups,
    max_rank=None,
    n_bootstrap=500,
    n_permutations=500,
    stability_floor=0.8,
    null_quantile=0.95,
    seed=0,
):
    """Trajectory-block bootstrap plus block-permutation parallel rank analysis.

    Returns the full spectra needed for an audit.  A component is retained only
    when its observed singular value beats its structured-null threshold and
    at least ``stability_floor`` of block-bootstrap spectra beat that same
    threshold.  Retained components must be consecutive from the leading one.
    """

    values = np.asarray(matrix, dtype=np.float64)
    labels = np.asarray(groups).reshape(-1)
    if values.ndim != 2 or values.size == 0 or len(labels) != len(values):
        raise ValueError("matrix must be nonempty and groups must align with rows")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix contains nonfinite values")
    boot_count = int(n_bootstrap)
    permutation_count = int(n_permutations)
    stability_cutoff = float(stability_floor)
    quantile = float(null_quantile)
    if boot_count < 1 or permutation_count < 1:
        raise ValueError("n_bootstrap and n_permutations must be positive")
    if not 0 <= stability_cutoff <= 1 or not 0 < quantile < 1:
        raise ValueError("stability_floor/null_quantile are outside valid ranges")
    unique = list(dict.fromkeys(labels.tolist()))
    if len(unique) < 2:
        raise ValueError("at least two trajectory groups are required")
    available = min(values.shape[1], max(len(values) - 1, 1))
    retained = available if max_rank is None else int(max_rank)
    if retained < 1:
        raise ValueError("max_rank must be positive")
    retained = min(retained, available)
    blocks = [np.flatnonzero(labels == group) for group in unique]

    def spectrum(sample):
        centered = sample - np.mean(sample, axis=0, keepdims=True)
        result = np.linalg.svd(centered, compute_uv=False)
        result = result / np.sqrt(max(len(sample) - 1, 1))
        padded = np.zeros(retained, dtype=np.float64)
        padded[: min(retained, len(result))] = result[:retained]
        return padded

    observed = spectrum(values)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((boot_count, retained), dtype=np.float64)
    for draw in range(boot_count):
        sampled = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[index] for index in sampled])
        bootstrap[draw] = spectrum(values[indices])

    # Independently exchange whole equal-length trajectory blocks by column.
    # Unequal singleton buckets use circular shifts; if those are degenerate,
    # a group-order permutation is applied to group means and residual blocks.
    buckets = {}
    for block in blocks:
        buckets.setdefault(len(block), []).append(block)
    null = np.empty((permutation_count, retained), dtype=np.float64)
    for draw in range(permutation_count):
        permuted = np.empty_like(values)
        for column in range(values.shape[1]):
            for bucket in buckets.values():
                if len(bucket) > 1:
                    order = rng.permutation(len(bucket))
                    for destination, source in enumerate(order):
                        permuted[bucket[destination], column] = values[
                            bucket[source], column
                        ]
                else:
                    block = bucket[0]
                    if len(block) > 1:
                        shift = int(rng.integers(0, len(block)))
                        permuted[block, column] = np.roll(values[block, column], shift)
                    else:
                        # Handled below if every block has length one.
                        permuted[block, column] = values[block, column]
            if all(len(block) == 1 for block in blocks):
                permuted[:, column] = values[rng.permutation(len(values)), column]
        null[draw] = spectrum(permuted)
    thresholds = np.quantile(null, quantile, axis=0)
    stability = np.mean(bootstrap > thresholds[None, :], axis=0)

    rank = 0
    for index in range(retained):
        if observed[index] <= thresholds[index] or stability[index] < stability_cutoff:
            break
        rank += 1
    rank_draws = np.zeros(boot_count, dtype=np.int64)
    for draw in range(boot_count):
        for index in range(retained):
            if bootstrap[draw, index] <= thresholds[index]:
                break
            rank_draws[draw] += 1
    rank_ci = tuple(
        int(value)
        for value in np.quantile(rank_draws, [0.025, 0.975], method="nearest")
    )
    return {
        "rank": int(rank),
        "selected_rank": int(rank),
        "singular_values": observed,
        "bootstrap_singular_values": bootstrap,
        "null_singular_values": null,
        "null_thresholds": thresholds,
        "rank_draws": rank_draws,
        "rank_ci95": rank_ci,
        "stability": stability,
        "selected_rank_stability": (
            float(stability[rank - 1]) if rank > 0 else 0.0
        ),
        "effective_rank": effective_rank(observed),
        "n_groups": len(unique),
    }


def fit_grouped_ridge(
    x,
    y,
    groups,
    penalties=(1e-6, 1e-4, 1e-2, 1.0),
    folds=5,
    seed=0,
):
    """Select a ridge penalty by whole-group OOF prediction and refit on all rows."""

    features = np.asarray(x, dtype=np.float64)
    target_raw = np.asarray(y, dtype=np.float64)
    labels = np.asarray(groups).reshape(-1)
    vector_target = target_raw.ndim == 1
    target = target_raw[:, None] if vector_target else target_raw
    if (
        features.ndim != 2
        or target.ndim != 2
        or len(features) != len(target)
        or len(labels) != len(features)
        or len(features) == 0
    ):
        raise ValueError("x, y, and groups must be nonempty and row-aligned")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(target)):
        raise ValueError("x and y must be finite")
    candidates = np.asarray(tuple(penalties), dtype=np.float64).reshape(-1)
    if (
        candidates.size == 0
        or not np.all(np.isfinite(candidates))
        or np.any(candidates < 0)
    ):
        raise ValueError("penalties must be finite, nonnegative, and nonempty")
    unique = np.asarray(list(dict.fromkeys(labels.tolist())), dtype=object)
    fold_count = int(folds)
    if fold_count < 2 or fold_count > len(unique):
        raise ValueError("folds must be between two and the number of groups")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    held_out_groups = np.array_split(shuffled, fold_count)

    def solve(train_x, train_y, penalty):
        mean_x = np.mean(train_x, axis=0)
        mean_y = np.mean(train_y, axis=0)
        centered_x = train_x - mean_x
        centered_y = train_y - mean_y
        if penalty == 0:
            weight = np.linalg.lstsq(centered_x, centered_y, rcond=None)[0]
        else:
            weight = np.linalg.solve(
                centered_x.T @ centered_x
                + float(penalty) * np.eye(centered_x.shape[1]),
                centered_x.T @ centered_y,
            )
        intercept = mean_y - mean_x @ weight
        return weight, intercept

    predictions = np.empty((len(candidates), *target.shape), dtype=np.float64)
    for fold_groups in held_out_groups:
        test = np.isin(labels, fold_groups)
        train = ~test
        if not np.any(train) or not np.any(test):
            raise ValueError("a grouped fold is empty")
        for index, penalty in enumerate(candidates):
            weight, intercept = solve(features[train], target[train], penalty)
            predictions[index, test] = features[test] @ weight + intercept
    mse = np.mean((predictions - target[None, :, :]) ** 2, axis=(1, 2))
    best_index = int(np.argmin(mse))
    best_penalty = float(candidates[best_index])
    weight, intercept = solve(features, target, best_penalty)
    oof = predictions[best_index]
    return {
        "weight": weight[:, 0] if vector_target else weight,
        "coef": weight[:, 0] if vector_target else weight,
        "intercept": float(intercept[0]) if vector_target else intercept,
        "penalty": best_penalty,
        "oof_prediction": oof[:, 0] if vector_target else oof,
        "oof_mse": float(mse[best_index]),
        "penalty_oof_mse": mse,
        "penalties": candidates,
    }


def fit_affine_bilinear_operator(states, actions, next_states, ridge=1e-6, modes=None):  # noqa: F811
    """Fit global or frozen-mode affine-bilinear dynamics and return plain dictionaries."""

    state = np.asarray(states, dtype=np.float64)
    action = np.asarray(actions, dtype=np.float64)
    target = np.asarray(next_states, dtype=np.float64)
    if (
        state.ndim != 2
        or action.ndim != 2
        or target.shape != state.shape
        or len(action) != len(state)
        or len(state) == 0
    ):
        raise ValueError("states, actions, and next_states must be nonempty and aligned")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)) or not np.all(np.isfinite(target)):
        raise ValueError("dynamics arrays must be finite")
    penalty = float(ridge)
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("ridge must be nonnegative and finite")
    if modes is not None:
        labels = np.asarray(modes).reshape(-1)
        if len(labels) != len(state):
            raise ValueError("modes must align with transitions")
        operators = {}
        for mode in dict.fromkeys(labels.tolist()):
            selected = labels == mode
            operators[mode] = fit_affine_bilinear_operator(
                state[selected], action[selected], target[selected], ridge=penalty
            )
        return {
            "operators": operators,
            "global_operator": fit_affine_bilinear_operator(
                state, action, target, ridge=penalty
            ),
            "mode_labels": tuple(operators),
        }
    interactions = [
        state * action[:, index : index + 1] for index in range(action.shape[1])
    ]
    design = np.column_stack(
        [np.ones(len(state), dtype=np.float64), state, action, *interactions]
    )
    if penalty == 0:
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    else:
        regularizer = penalty * np.eye(design.shape[1], dtype=np.float64)
        regularizer[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + regularizer, design.T @ target
        )
    state_dim = state.shape[1]
    action_dim = action.shape[1]
    action_start = 1 + state_dim
    interaction_start = action_start + action_dim
    operators = np.stack(
        [
            coefficients[
                interaction_start + index * state_dim : interaction_start
                + (index + 1) * state_dim
            ].T
            for index in range(action_dim)
        ],
        axis=0,
    )
    fitted = design @ coefficients
    return {
        "c": coefficients[0],
        "intercept": coefficients[0],
        "A": coefficients[1:action_start].T,
        "B": coefficients[action_start:interaction_start].T,
        "N": operators,
        "ridge": penalty,
        "n_samples": len(state),
        "training_rmse": float(np.sqrt(np.mean((fitted - target) ** 2))),
    }


def predict_affine_bilinear(operator, states, actions, modes=None):
    """Predict dict- or dataclass-backed global/frozen-mode affine-bilinear dynamics."""

    if isinstance(operator, dict) and "operators" in operator:
        if modes is None:
            raise ValueError("mode-conditioned prediction requires modes")
        state_raw = np.asarray(states, dtype=np.float64)
        single = state_raw.ndim == 1
        state = state_raw[None, :] if single else state_raw
        labels = np.asarray(modes)
        labels = np.repeat(labels.reshape(1), len(state)) if labels.ndim == 0 else labels.reshape(-1)
        action_raw = np.asarray(actions, dtype=np.float64)
        if action_raw.ndim == 0:
            action = np.full((len(state), 1), float(action_raw))
        elif action_raw.ndim == 1 and single:
            action = action_raw[None, :]
        elif action_raw.ndim == 1:
            action_dim = next(iter(operator["operators"].values()))["B"].shape[1]
            action = action_raw[:, None] if action_dim == 1 and len(action_raw) == len(state) else np.broadcast_to(action_raw, (len(state), action_dim))
        else:
            action = action_raw
        if len(labels) != len(state) or len(action) != len(state):
            raise ValueError("states, actions, and modes are not aligned")
        result = np.empty_like(state)
        for row, mode in enumerate(labels.tolist()):
            if mode not in operator["operators"]:
                raise ValueError(f"unknown mode {mode!r}")
            result[row] = predict_affine_bilinear(
                operator["operators"][mode], state[row], action[row]
            )
        return result[0] if single else result

    get = operator.get if isinstance(operator, dict) else lambda key, default=None: getattr(operator, key, default)
    c = np.asarray(get("c", get("intercept")), dtype=np.float64)
    A = np.asarray(get("A"), dtype=np.float64)
    B = np.asarray(get("B"), dtype=np.float64)
    N = np.asarray(get("N"), dtype=np.float64)
    state_raw = np.asarray(states, dtype=np.float64)
    action_raw = np.asarray(actions, dtype=np.float64)
    single = state_raw.ndim == 1
    state = state_raw[None, :] if single else state_raw
    if action_raw.ndim == 0:
        if B.shape[1] != 1:
            raise ValueError("a scalar action requires action dimension one")
        action = np.full((len(state), 1), float(action_raw))
    elif action_raw.ndim == 1:
        if single:
            action = action_raw[None, :]
        elif B.shape[1] == 1 and len(action_raw) == len(state):
            action = action_raw[:, None]
        elif len(action_raw) == B.shape[1]:
            action = np.broadcast_to(action_raw, (len(state), B.shape[1]))
        else:
            raise ValueError("one-dimensional actions are ambiguous")
    else:
        action = action_raw
    if state.ndim != 2 or state.shape[1] != A.shape[0] or action.shape != (len(state), B.shape[1]):
        raise ValueError("state/action dimensions do not match the operator")
    result = c + state @ A.T + action @ B.T
    for index in range(B.shape[1]):
        result += action[:, index : index + 1] * (state @ N[index].T)
    return result[0] if single else result


def compose_affine_bilinear(
    model,
    initial_states,
    action_sequences=None,
    mode_sequences=None,
):
    """Apply an affine-bilinear model to batched multi-step action words.

    ``initial_states`` may be one state or a row batch.  Actions may be one
    shared ``(horizon, action_dim)`` word or one word per initial state.  Frozen
    mode sequences follow the same shared/batched convention.  For backwards
    compatibility, omitting ``action_sequences`` treats ``initial_states`` as
    an action word and returns its composed ``(matrix, offset)``.
    """

    hybrid = isinstance(model, dict) and "operators" in model
    exemplar = model["global_operator"] if hybrid else model
    get_exemplar = exemplar.get if isinstance(exemplar, dict) else lambda key, default=None: getattr(exemplar, key, default)
    A0 = np.asarray(get_exemplar("A"), dtype=np.float64)
    B0 = np.asarray(get_exemplar("B"), dtype=np.float64)
    if action_sequences is None:
        sequence = np.asarray(initial_states, dtype=np.float64)
        if sequence.ndim == 1:
            sequence = sequence[:, None] if B0.shape[1] == 1 else sequence[None, :]
        if sequence.ndim != 2 or sequence.shape[1] != B0.shape[1]:
            raise ValueError("actions must be horizon by action dimension")
        labels = (
            np.asarray([None] * len(sequence), dtype=object)
            if not hybrid
            else np.asarray(mode_sequences).reshape(-1)
        )
        if hybrid and len(labels) != len(sequence):
            raise ValueError("hybrid composition requires one mode per action")
        matrix = np.eye(A0.shape[0], dtype=np.float64)
        offset = np.zeros(A0.shape[0], dtype=np.float64)
        for action, mode in zip(sequence, labels.tolist()):
            current = model["operators"][mode] if hybrid else model
            get = current.get if isinstance(current, dict) else lambda key, default=None: getattr(current, key, default)
            A = np.asarray(get("A"), dtype=np.float64)
            B = np.asarray(get("B"), dtype=np.float64)
            N = np.asarray(get("N"), dtype=np.float64)
            c = np.asarray(get("c", get("intercept")), dtype=np.float64)
            step_matrix = A + np.tensordot(action, N, axes=(0, 0))
            step_offset = c + B @ action
            offset = step_matrix @ offset + step_offset
            matrix = step_matrix @ matrix
        return matrix, offset

    state_raw = np.asarray(initial_states, dtype=np.float64)
    single = state_raw.ndim == 1
    states = state_raw[None, :] if single else state_raw
    actions = np.asarray(action_sequences, dtype=np.float64)
    if actions.ndim == 1:
        actions = actions[:, None] if B0.shape[1] == 1 else actions[None, :]
    if actions.ndim == 2:
        actions = np.broadcast_to(actions, (len(states), *actions.shape))
    if (
        states.ndim != 2
        or states.shape[1] != A0.shape[0]
        or actions.ndim != 3
        or actions.shape[0] != len(states)
        or actions.shape[2] != B0.shape[1]
    ):
        raise ValueError("initial states and action sequences have incompatible shapes")
    if hybrid:
        if mode_sequences is None:
            raise ValueError("hybrid composition requires frozen mode sequences")
        modes = np.asarray(mode_sequences)
        if modes.ndim == 1:
            modes = np.broadcast_to(modes, (len(states), len(modes)))
        if modes.shape != actions.shape[:2]:
            raise ValueError("mode sequences must match batch and horizon")
    else:
        modes = np.full(actions.shape[:2], None, dtype=object)
    current_states = states.copy()
    for step in range(actions.shape[1]):
        if hybrid:
            next_states = np.empty_like(current_states)
            for row in range(len(states)):
                mode = modes[row, step].item() if isinstance(modes[row, step], np.generic) else modes[row, step]
                if mode not in model["operators"]:
                    raise ValueError(f"unknown mode {mode!r}")
                next_states[row] = predict_affine_bilinear(
                    model["operators"][mode], current_states[row], actions[row, step]
                )
            current_states = next_states
        else:
            current_states = predict_affine_bilinear(
                model, current_states, actions[:, step, :]
            )
    return current_states[0] if single else current_states


def fit_whitened_similarity(
    source,
    target,
    max_condition=100.0,
    min_singular_value=1e-3,
    regularization=1e-10,
):
    """Fit one invertible affine whitened-Procrustes similarity map."""

    first = np.asarray(source, dtype=np.float64)
    second = np.asarray(target, dtype=np.float64)
    if first.ndim != 2 or first.shape != second.shape or len(first) < 2:
        raise ValueError("source and target must have the same nontrivial 2-D shape")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("source and target must be finite")
    ridge = float(regularization)
    limit = float(max_condition)
    minimum = float(min_singular_value)
    if (
        not np.isfinite(ridge)
        or ridge < 0
        or not np.isfinite(limit)
        or limit <= 1
        or not np.isfinite(minimum)
        or minimum <= 0
    ):
        raise ValueError("regularization/condition checks are invalid")
    first_mean = np.mean(first, axis=0)
    second_mean = np.mean(second, axis=0)
    first_centered = first - first_mean
    second_centered = second - second_mean
    divisor = max(len(first) - 1, 1)
    source_chart_singular = np.linalg.svd(
        first_centered / np.sqrt(divisor), compute_uv=False
    )
    target_chart_singular = np.linalg.svd(
        second_centered / np.sqrt(divisor), compute_uv=False
    )
    if (
        len(source_chart_singular) < first.shape[1]
        or len(target_chart_singular) < first.shape[1]
        or source_chart_singular[-1] < minimum
        or target_chart_singular[-1] < minimum
    ):
        raise ValueError("calibration chart is rank deficient below min_singular_value")
    covariance_first = first_centered.T @ first_centered / divisor + ridge * np.eye(first.shape[1])
    covariance_second = second_centered.T @ second_centered / divisor + ridge * np.eye(first.shape[1])

    def root(matrix, inverse):
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        if np.any(eigenvalues <= 0):
            raise ValueError("regularized covariance is not positive definite")
        power = 1.0 / np.sqrt(eigenvalues) if inverse else np.sqrt(eigenvalues)
        return (eigenvectors * power) @ eigenvectors.T, eigenvalues

    first_inverse_root, first_eigenvalues = root(covariance_first, True)
    second_inverse_root, second_eigenvalues = root(covariance_second, True)
    second_root, _ = root(covariance_second, False)
    first_condition = float(first_eigenvalues[-1] / first_eigenvalues[0])
    second_condition = float(second_eigenvalues[-1] / second_eigenvalues[0])
    whitened_first = first_centered @ first_inverse_root
    whitened_second = second_centered @ second_inverse_root
    left, _, right_t = np.linalg.svd(
        whitened_first.T @ whitened_second, full_matrices=False
    )
    row_map = first_inverse_root @ (left @ right_t) @ second_root
    matrix = row_map.T
    condition = float(np.linalg.cond(matrix))
    map_singular_values = np.linalg.svd(matrix, compute_uv=False)
    if (
        not np.isfinite(condition)
        or condition > limit
        or map_singular_values[-1] < minimum
    ):
        raise ValueError("fitted map is singular or exceeds max_condition")
    inverse = np.linalg.inv(matrix)
    offset = second_mean - matrix @ first_mean
    prediction = first @ matrix.T + offset
    denominator = max(
        float(np.sqrt(np.mean(second_centered**2))), np.finfo(np.float64).tiny
    )
    return {
        "matrix": matrix,
        "inverse": inverse,
        "offset": offset,
        "source_mean": first_mean,
        "target_mean": second_mean,
        "condition_number": condition,
        "source_covariance_condition": first_condition,
        "target_covariance_condition": second_condition,
        "minimum_singular_value": float(map_singular_values[-1]),
        "calibration_rmse": float(np.sqrt(np.mean((prediction - second) ** 2))),
        "calibration_relative_rmse": float(
            np.sqrt(np.mean((prediction - second) ** 2)) / denominator
        ),
    }


def operator_intertwining_metrics(source, target, transport=None, reference=None):  # noqa: F811
    """Prediction-level agreement or direct affine-bilinear intertwining errors."""

    if not isinstance(source, dict) and not hasattr(source, "A"):
        first = np.asarray(source, dtype=np.float64)
        second = np.asarray(target, dtype=np.float64)
        if first.size == 0 or first.shape != second.shape:
            raise ValueError("left and right predictions must have the same nonempty shape")
        if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
            raise ValueError("predictions must be finite")
        residual = first - second
        rmse = float(np.sqrt(np.mean(residual**2)))
        prediction_reference = transport if reference is None and transport is not None else reference
        denominator_values = (
            second
            if prediction_reference is None
            else np.asarray(prediction_reference, dtype=np.float64)
        )
        if denominator_values.shape != second.shape or not np.all(np.isfinite(denominator_values)):
            raise ValueError("reference must be finite and match the predictions")
        denominator = max(
            float(np.sqrt(np.mean(denominator_values**2))),
            np.finfo(np.float64).tiny,
        )
        first_flat = first.reshape(-1)
        second_flat = second.reshape(-1)
        norm_product = float(np.linalg.norm(first_flat) * np.linalg.norm(second_flat))
        cosine = (
            float(np.dot(first_flat, second_flat) / norm_product)
            if norm_product > np.finfo(np.float64).tiny
            else 0.0
        )
        return {
            "rmse": rmse,
            "relative_rmse": float(rmse / denominator),
            "cosine": float(np.clip(cosine, -1.0, 1.0)),
            "residual": residual,
        }

    if transport is None:
        raise ValueError("operator intertwining requires a transport map")

    source_get = source.get if isinstance(source, dict) else lambda key, default=None: getattr(source, key, default)
    target_get = target.get if isinstance(target, dict) else lambda key, default=None: getattr(target, key, default)
    source_c = np.asarray(source_get("c", source_get("intercept")), dtype=np.float64)
    source_A = np.asarray(source_get("A"), dtype=np.float64)
    source_B = np.asarray(source_get("B"), dtype=np.float64)
    source_N = np.asarray(source_get("N"), dtype=np.float64)
    target_c = np.asarray(target_get("c", target_get("intercept")), dtype=np.float64)
    target_A = np.asarray(target_get("A"), dtype=np.float64)
    target_B = np.asarray(target_get("B"), dtype=np.float64)
    target_N = np.asarray(target_get("N"), dtype=np.float64)
    if isinstance(transport, dict):
        matrix = np.asarray(transport["matrix"], dtype=np.float64)
        offset = np.asarray(transport.get("offset", np.zeros(len(matrix))), dtype=np.float64)
    elif hasattr(transport, "matrix"):
        matrix = np.asarray(transport.matrix, dtype=np.float64)
        offset = np.asarray(transport.offset, dtype=np.float64)
    else:
        matrix = np.asarray(transport, dtype=np.float64)
        offset = np.zeros(len(matrix), dtype=np.float64)
    if matrix.shape != source_A.shape or target_A.shape != source_A.shape:
        raise ValueError("transport and operator state dimensions differ")
    A_left, A_right = target_A @ matrix, matrix @ source_A
    N_left = np.einsum("uij,jk->uik", target_N, matrix)
    N_right = np.einsum("ij,ujk->uik", matrix, source_N)
    B_left = target_B.copy()
    for index in range(target_B.shape[1]):
        B_left[:, index] += target_N[index] @ offset
    B_right = matrix @ source_B
    c_left = target_c + target_A @ offset
    c_right = matrix @ source_c + offset

    def relative(left, right):
        denominator = max(
            float(np.linalg.norm(left)),
            float(np.linalg.norm(right)),
            np.finfo(np.float64).tiny,
        )
        return float(np.linalg.norm(left - right) / denominator)

    residuals = [A_left - A_right, B_left - B_right, N_left - N_right, c_left - c_right]
    references = [A_left, B_left, N_left, c_left, A_right, B_right, N_right, c_right]
    numerator = np.sqrt(sum(float(np.sum(value**2)) for value in residuals))
    denominator = np.sqrt(0.5 * sum(float(np.sum(value**2)) for value in references))
    return {
        "aggregate_relative_error": float(
            numerator / max(denominator, np.finfo(np.float64).tiny)
        ),
        "A_relative_error": relative(A_left, A_right),
        "B_relative_error": relative(B_left, B_right),
        "N_relative_error": relative(N_left, N_right),
        "intercept_relative_error": relative(c_left, c_right),
    }


def reachability_observability_diagnostics(
    a_matrices,
    b_matrices,
    c_matrix=None,
    tolerance=1e-8,
    reference_state=None,
    reference_action=None,
    horizon=None,
):
    """Reachability/observability diagnostics for one or several frozen modes."""

    operator_input = isinstance(a_matrices, dict) or hasattr(a_matrices, "A")
    if operator_input:
        operator = a_matrices
        get = operator.get if isinstance(operator, dict) else lambda key, default=None: getattr(operator, key, default)
        A_values = np.asarray(get("A"), dtype=np.float64)[None, :, :]
        B_values = np.asarray(get("B"), dtype=np.float64)[None, :, :]
        N = np.asarray(get("N"), dtype=np.float64)
        C = np.asarray(b_matrices if c_matrix is None else c_matrix, dtype=np.float64)
    else:
        A_values = np.asarray(a_matrices, dtype=np.float64)
        B_values = np.asarray(b_matrices, dtype=np.float64)
        if A_values.ndim == 2:
            A_values = A_values[None, :, :]
        if B_values.ndim == 2:
            B_values = B_values[None, :, :]
        if len(A_values) != len(B_values):
            if len(B_values) == 1:
                B_values = np.broadcast_to(B_values, (len(A_values), *B_values.shape[1:]))
            else:
                raise ValueError("a_matrices and b_matrices have different mode counts")
        N = None
        C = np.eye(A_values.shape[1]) if c_matrix is None else np.asarray(c_matrix, dtype=np.float64)
    if A_values.ndim != 3 or B_values.ndim != 3:
        raise ValueError("A and B inputs must be matrices or mode stacks")
    A = A_values[0]
    B = B_values[0]
    if (
        A.ndim != 2
        or A.shape[0] != A.shape[1]
        or B.shape[0] != len(A)
        or (operator_input and N.shape != (B.shape[1], len(A), len(A)))
    ):
        raise ValueError("operator matrices have inconsistent dimensions")
    if C.ndim != 2 or C.shape[1] != len(A):
        raise ValueError("observation_matrix has the wrong dimension")
    steps = len(A) if horizon is None else int(horizon)
    if steps < 1:
        raise ValueError("horizon must be positive")
    def diagnose(matrix):
        singular = np.linalg.svd(matrix, compute_uv=False)
        threshold = (
            max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
            if tolerance is None else float(tolerance)
        )
        rank = int(np.sum(singular > threshold))
        full = rank == min(matrix.shape)
        condition = float(singular[0] / singular[-1]) if full and singular[-1] > 0 else np.inf
        return {
            "matrix": matrix,
            "singular_values": singular,
            "rank": rank,
            "full_rank": bool(full),
            "condition_number": condition,
            "effective_rank": effective_rank(singular),
        }

    by_mode = []
    for mode_index, (A, B) in enumerate(zip(A_values, B_values)):
        if A.shape != (A_values.shape[1], A_values.shape[1]) or B.shape[0] != len(A):
            raise ValueError("operator matrices have inconsistent dimensions")
        if operator_input:
            state = np.zeros(len(A)) if reference_state is None else np.asarray(reference_state, dtype=np.float64).reshape(-1)
            action = np.zeros(B.shape[1]) if reference_action is None else np.asarray(reference_action, dtype=np.float64).reshape(-1)
            if len(state) != len(A) or len(action) != B.shape[1]:
                raise ValueError("reference state/action dimensions are wrong")
            transition = A + np.tensordot(action, N, axes=(0, 0))
            input_jacobian = B.copy()
            for index in range(B.shape[1]):
                input_jacobian[:, index] += N[index] @ state
        else:
            transition = A
            input_jacobian = B
        reachable = []
        observable = []
        power = np.eye(len(A))
        for _ in range(steps):
            reachable.append(power @ input_jacobian)
            observable.append(C @ power)
            power = transition @ power
        reachability = np.column_stack(reachable)
        observability = np.vstack(observable)
        by_mode.append(
            {
                "mode_index": mode_index,
                "reachability": diagnose(reachability),
                "observability": diagnose(observability),
                "transition_matrix": transition,
                "input_matrix": input_jacobian,
            }
        )
    reachability_ranks = np.asarray([row["reachability"]["rank"] for row in by_mode], dtype=np.int64)
    observability_ranks = np.asarray([row["observability"]["rank"] for row in by_mode], dtype=np.int64)
    return {
        "by_mode": by_mode,
        "reachability_ranks": reachability_ranks,
        "observability_ranks": observability_ranks,
        "reachability_rank": int(np.min(reachability_ranks)),
        "observability_rank": int(np.min(observability_ranks)),
        "all_modes_reachable": bool(np.all(reachability_ranks == A_values.shape[1])),
        "all_modes_observable": bool(np.all(observability_ranks == A_values.shape[1])),
    }


def interchange_metrics(
    observed_effects,
    target_effects,
    baseline_errors=None,
    patched_errors=None,
    minimum_effect_energy=1e-12,
):
    """Effect transport cosine/error plus optional rowwise counterfactual error gain."""

    observed = np.asarray(observed_effects, dtype=np.float64)
    intended = np.asarray(target_effects, dtype=np.float64)
    if observed.size == 0 or observed.shape != intended.shape:
        raise ValueError("observed and target effects must have the same nonempty shape")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(intended)):
        raise ValueError("interchange arrays must be finite")
    floor = float(minimum_effect_energy)
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("minimum_effect_energy must be positive and finite")
    intended_energy = float(np.sum(intended**2))
    observed_energy = float(np.sum(observed**2))
    if intended_energy < floor:
        return {
            "eligible": False,
            "intended_effect_energy": intended_energy,
            "observed_effect_energy": observed_energy,
            "effect_cosine": np.nan,
            "cosine": np.nan,
            "relative_effect_error": np.nan,
            "relative_error": np.nan,
            "bounded_effect_error": np.nan,
            "error_gain": np.nan,
            "row_relative_error": np.full(observed.shape[0] if observed.ndim > 1 else 1, np.nan),
        }
    intended_norm = np.sqrt(intended_energy)
    observed_norm = np.sqrt(observed_energy)
    error = float(np.linalg.norm(observed - intended))
    relative_error = error / intended_norm
    cosine = (
        float(np.sum(observed * intended) / (observed_norm * intended_norm))
        if observed_norm > np.finfo(np.float64).tiny
        else 0.0
    )
    observed_rows = observed.reshape(1, -1) if observed.ndim == 1 else observed.reshape(len(observed), -1)
    intended_rows = intended.reshape(1, -1) if intended.ndim == 1 else intended.reshape(len(intended), -1)
    row_error = np.linalg.norm(observed_rows - intended_rows, axis=1)
    row_target_norm = np.linalg.norm(intended_rows, axis=1)
    row_relative = np.divide(
        row_error,
        row_target_norm,
        out=np.full_like(row_error, np.nan),
        where=row_target_norm**2 >= floor,
    )
    if baseline_errors is None or patched_errors is None:
        row_gain = np.full(len(row_relative), np.nan)
        error_gain = np.nan
    else:
        baseline = np.asarray(baseline_errors, dtype=np.float64).reshape(-1)
        patched = np.asarray(patched_errors, dtype=np.float64).reshape(-1)
        if baseline.shape != patched.shape or len(baseline) != len(row_relative) or not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(patched)):
            raise ValueError("baseline_errors and patched_errors must align with effect rows")
        row_gain = baseline - patched
        error_gain = float(np.mean(row_gain))
    return {
        "eligible": True,
        "intended_effect_energy": intended_energy,
        "observed_effect_energy": observed_energy,
        "effect_cosine": float(np.clip(cosine, -1.0, 1.0)),
        "cosine": float(np.clip(cosine, -1.0, 1.0)),
        "relative_effect_error": float(relative_error),
        "relative_error": float(relative_error),
        "bounded_effect_error": float(relative_error / (1.0 + relative_error)),
        "error_gain": error_gain,
        "row_relative_error": row_relative,
        "row_error_gain": row_gain,
    }


def clustered_bootstrap_interval(  # noqa: F811
    values,
    groups,
    state_groups=None,
    draws=2000,
    seed=0,
    alpha=0.05,
    n_bootstrap=None,
    confidence=None,
):
    """Mean percentile interval resampling trajectories and optional state groups."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    trajectories = np.asarray(groups).reshape(-1)
    if array.size == 0 or len(array) != len(trajectories) or not np.all(np.isfinite(array)):
        raise ValueError("values and groups must be finite, nonempty, and aligned")
    states = None if state_groups is None else np.asarray(state_groups).reshape(-1)
    if states is not None and len(states) != len(array):
        raise ValueError("state_groups must align with values")
    repetitions = int(draws if n_bootstrap is None else n_bootstrap)
    tail = float(alpha if confidence is None else 1.0 - confidence)
    level = 1.0 - tail
    if repetitions < 1 or not 0 < level < 1:
        raise ValueError("n_bootstrap/confidence are invalid")
    rng = np.random.default_rng(seed)
    trajectory_units = list(dict.fromkeys(trajectories.tolist()))
    state_units = None if states is None else list(dict.fromkeys(states.tolist()))
    top_count = len(trajectory_units) if states is None else len(state_units)
    if top_count < 2:
        raise ValueError("at least two independent clusters are required")
    draws = np.empty(repetitions, dtype=np.float64)
    for draw in range(repetitions):
        pieces = []
        if states is None:
            sampled = rng.integers(0, len(trajectory_units), size=len(trajectory_units))
            pieces = [
                np.flatnonzero(trajectories == trajectory_units[index])
                for index in sampled
            ]
        else:
            sampled_states = rng.integers(0, len(state_units), size=len(state_units))
            for state_index in sampled_states:
                state = state_units[state_index]
                in_state = states == state
                local = list(dict.fromkeys(trajectories[in_state].tolist()))
                sampled_trajectories = rng.integers(0, len(local), size=len(local))
                pieces.extend(
                    np.flatnonzero(in_state & (trajectories == local[index]))
                    for index in sampled_trajectories
                )
        indices = np.concatenate(pieces)
        draws[draw] = float(np.mean(array[indices]))
    low, high = np.quantile(draws, [tail / 2.0, 1.0 - tail / 2.0])
    return [float(low), float(high)]


def holm_adjust(p_values, alpha=0.05):
    """Return Holm-adjusted p-values and step-down rejection decisions."""

    values = np.asarray(p_values, dtype=np.float64)
    shape = values.shape
    flat = values.reshape(-1)
    if flat.size == 0 or not np.all(np.isfinite(flat)) or np.any((flat < 0) | (flat > 1)):
        raise ValueError("p_values must be finite, nonempty, and lie in [0, 1]")
    level = float(alpha)
    if not 0 < level < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    order = np.argsort(flat, kind="stable")
    sorted_values = flat[order]
    factors = np.arange(len(flat), 0, -1, dtype=np.float64)
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(factors * sorted_values))
    adjusted = np.empty_like(flat)
    adjusted[order] = adjusted_sorted
    adjusted = adjusted.reshape(shape)
    return {
        "adjusted_pvalues": adjusted,
        "adjusted_p_values": adjusted,
        "reject": adjusted <= level,
        "alpha": level,
    }


def derive_decision(
    gates=None,
    run_mode="pilot",
    confirmation_eligible=True,
    **named_gates,
):
    """Derive pass/partial-pass/fail from cumulative Stage 33 gate booleans."""

    supplied = {} if gates is None else dict(gates)
    supplied.update(named_gates)
    aliases = {
        "stable_low_rank": ("stable_low_rank", "rank", "rank_pass"),
        "hybrid_improvement": ("hybrid_improvement", "hybrid", "hybrid_pass"),
        "fixed_map_well_conditioned": ("fixed_map_well_conditioned", "fixed_map", "map_pass"),
        "heldout_conjugacy": ("heldout_conjugacy", "conjugacy", "conjugacy_pass"),
        "internal_interchange": ("internal_interchange", "interchange", "interchange_pass"),
        "planning_value": ("planning_value", "planning", "planning_pass"),
        "controls_rejected": ("controls_rejected", "controls", "controls_pass"),
        "family_consistency": ("family_consistency", "families", "family_pass"),
    }
    checks = {}
    for canonical, alternatives in aliases.items():
        present = [name for name in alternatives if name in supplied]
        if present:
            checks[canonical] = bool(supplied[present[0]])
        elif canonical == "family_consistency":
            checks[canonical] = True
        else:
            checks[canonical] = False
    mode = str(run_mode)
    if mode not in {"smoke", "pilot"}:
        raise ValueError("run_mode must be 'smoke' or 'pilot'")
    eligible = bool(confirmation_eligible) and mode == "pilot"
    passed = eligible and all(checks.values())
    partial = (
        checks["stable_low_rank"]
        and checks["hybrid_improvement"]
        and checks["fixed_map_well_conditioned"]
        and checks["heldout_conjugacy"]
        and checks["controls_rejected"]
        and checks["family_consistency"]
    )
    status = "pass" if passed else "partial_pass" if eligible and partial else "fail"
    evidence_level = (
        6
        if passed
        else 5
        if partial and checks["internal_interchange"]
        else 4
        if partial
        else 3
        if checks["stable_low_rank"]
        else 1
    )
    return {
        "status": status,
        "passed": bool(passed),
        "evidence_level": evidence_level,
        "level": evidence_level,
        "run_mode": mode,
        "confirmation_eligible": eligible,
        "checks": checks,
        "failed_checks": tuple(name for name, value in checks.items() if not value),
    }


__all__ += [
    "compose_affine_bilinear",
    "derive_decision",
    "fit_grouped_ridge",
    "fit_whitened_similarity",
    "holm_adjust",
    "interchange_metrics",
    "reachability_observability_diagnostics",
    "select_stable_rank",
    "signature_pseudometric",
]
