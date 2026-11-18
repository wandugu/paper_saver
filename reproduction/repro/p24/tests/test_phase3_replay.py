import numpy as np

from repro.p24.phase3_replay import Population, agreement, auroc, calibrate, paired, policy_family, select


def population():
    records = [{'id': str(index), 'tokens': ['same' if index < 2 else str(index)], 'images': ['a.jpg']}
               for index in range(4)]
    rows = [{'id': str(index), 'gold': [1], 'text_prediction': [0], 'vision_prediction': [1],
             'gate': [.9 - .2 * index], 'eligible': [True], 'selected': [[0]]} for index in range(4)]
    return Population(rows, records, {'off_ms': 10., 'on_ms': 20.})


def test_distinct_cost_deduplicates_shared_attachment():
    metric = population().evaluate({'threshold': 0., 'eta': 0.})
    assert metric['detail'] == 1
    assert metric['coverage'] == 1
    assert metric['f1'] == 1
    assert metric['latency_ms'] == 20


def test_threshold_equality_and_exact_off():
    data = population()
    assert data.active({'threshold': .9, 'eta': 0.}).tolist() == [True, False, False, False]
    curve = calibrate(data, policy_family([0]))
    assert curve[-1]['ucb'] == 0
    assert curve[-1]['f1'] == 0
    assert all(row['ucb'] > .1 for row in curve[:-1])


def test_small_calibration_falls_back_without_fabricating_bound():
    data = population()
    chosen, fallback = select(calibrate(data, policy_family([0])), data, 'value_distinct', .1)
    assert chosen['id'] == 'off' and fallback


def test_value_selection_can_choose_off_without_fallback():
    data = population()
    curve = calibrate(data, policy_family([0]))
    for row in curve:
        row['feasible'] = True
    chosen, fallback = select(curve, data, 'value_distinct', 2.)
    assert chosen['id'] == 'off' and not fallback


def test_auroc_uses_tie_aware_ranks():
    assert auroc([0, 1], [False, True]) == 1
    assert auroc([1, 1], [False, True]) == .5
    assert auroc([1, 1], [False, False]) is None


def test_family_sizes_and_penalty_score():
    assert len(policy_family([0, .5, 1])) == 304
    data = population()
    assert np.all(data.active({'threshold': .8, 'eta': 1.}) == [True, False, False, False])


def test_paired_identical_predictions_have_zero_interval():
    data = population()
    result = paired(data, data.text, data.text, 42, bootstrap=20)
    assert result['f1_delta'] == result['ci_low'] == result['ci_high'] == 0
    assert result['post_disagreement'] == 0


def test_post_agreement_reports_degenerate_labels():
    result = agreement(population())
    assert result['positive_posts'] == 3
    assert result['post_auroc'] is None
