import numpy as np

from cf_faithfulness.stage34_action_specificity_repair import (
    Stage341Gates,
    action_blind_context_features,
    action_prefix_features,
    action_response_path_rows,
    clustered_bootstrap_interval,
    deranged_word_rows,
    derive_stage341_decision,
    fit_grouped_rff_ridge,
    grouped_record_mse,
    predict_grouped_rff_ridge,
    relative_advantage,
)


def _toy_paths(offset=0.0):
    names = ["AA", "BB", "zero2"]
    lengths = np.array([2, 2, 2])
    paths = np.zeros((3, 2, 2), dtype=np.float64) + float(offset)
    paths[0] += [[1.0, 0.0], [2.0, 0.0]]
    paths[1] += [[0.0, 1.0], [0.0, 2.0]]
    return paths, names, lengths


def test_path_rows_remove_static_offset_without_action_indexed_columns():
    paths, names, lengths = _toy_paths()
    shifted, _, _ = _toy_paths(19.0)
    args = (names, lengths, ["AA", "BB"], {2: "zero2"})
    rows, metadata = action_response_path_rows(paths, *args)
    shifted_rows, shifted_metadata = action_response_path_rows(shifted, *args)
    assert rows.shape == (4, 2)
    assert np.array_equal(rows, shifted_rows)
    assert np.array_equal(metadata["word"], shifted_metadata["word"])
    assert rows.shape[1] == paths.shape[2]


def test_action_blind_context_is_identical_for_different_words():
    state = np.repeat(np.array([[1.0, 2.0, 3.0]]), 4, axis=0)
    features = action_blind_context_features(
        state,
        [2, 2, 2, 2],
        [1, 2, 1, 2],
        ["free"] * 4,
        ["free", "contact"],
        maximum_length=2,
    )
    assert np.array_equal(features[0], features[2])
    assert np.array_equal(features[1], features[3])


def test_derangement_swaps_whole_same_length_paths():
    paths, names, lengths = _toy_paths()
    rows, metadata = action_response_path_rows(
        paths, names, lengths, ["AA", "BB"], {2: "zero2"}
    )
    shuffled = deranged_word_rows(rows, metadata, seed=3)
    assert np.array_equal(shuffled[:2], rows[2:])
    assert np.array_equal(shuffled[2:], rows[:2])


def test_action_prefix_features_align_with_response_rows():
    names = ["AA", "BB"]
    lengths = np.array([2, 2])
    actions = np.zeros((2, 4, 2), dtype=np.float64)
    actions[0, :, 0] = 1.0
    actions[1, :, 1] = 2.0
    features = action_prefix_features(
        actions,
        np.ones((2, 4), dtype=bool),
        names,
        names,
        lengths,
        frameskip=2,
    )
    assert features.shape == (4, 7)
    assert np.allclose(features[0, :2], [2.0, 0.0])
    assert np.allclose(features[3, :2], [0.0, 8.0])


def test_grouped_rff_and_record_metrics_are_deterministic():
    rng = np.random.default_rng(34101)
    groups = np.repeat(np.arange(12), 6)
    x = rng.normal(size=(len(groups), 4))
    y = np.column_stack([x[:, 0] + x[:, 1], x[:, 2] - x[:, 3]])
    model = fit_grouped_rff_ridge(
        x, y, groups, width=64, penalties=(1e-4, 1e-2), seed=9
    )
    prediction = predict_grouped_rff_ridge(model, x)
    error, ids = grouped_record_mse(prediction, y, groups)
    assert len(error) == len(ids) == 12
    assert np.mean(error) < 0.05
    assert np.allclose(
        predict_grouped_rff_ridge(model, x),
        predict_grouped_rff_ridge(model, x),
    )


def test_advantage_bootstrap_and_bounded_decision():
    primary = np.array([1.0, 1.0, 1.0, 1.0])
    control = np.array([2.0, 2.0, 2.0, 2.0])
    advantage = relative_advantage(primary, control)
    interval = clustered_bootstrap_interval(
        advantage, [0, 0, 1, 1], draws=100, seed=7
    )
    assert np.allclose(advantage, 0.5)
    assert np.allclose(interval, (0.5, 0.5))

    passed = derive_stage341_decision(Stage341Gates(True, True, True, True, True))
    assert passed["status"] == "action_specificity_repaired_continue_stage34"
    assert passed["passed"]
    failed = derive_stage341_decision(Stage341Gates(True, False, True, True, True))
    assert failed["status"] == "action_specificity_not_established"
    assert not failed["passed"]
