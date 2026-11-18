import math

import pytest

from repro.p24.phase2_audit import (calibrate_posts, post_metrics, routed_rows,
                                    validate_population)


def population():
    records = [{'id': str(index), 'tokens': [str(index // 2)], 'images': ['photo'], 'split': 'train'}
               for index in range(4)]
    rows = [{'id': str(index), 'gold': [1], 'text_prediction': [int(index == 0)],
             'vision_prediction': [int(index == 1)], 'gate': [.9], 'eligible': [True]}
            for index in range(4)]
    return records, rows


def test_positive_net_post_loss_does_not_count_cancelled_pair_flips():
    records, rows = population()
    metrics = post_metrics(routed_rows(rows, .5), records)
    assert metrics['post_harm'] == 0
    assert metrics['harmful_posts'] == 0
    rows[1]['vision_prediction'] = [0]
    metrics = post_metrics(routed_rows(rows, .5), records)
    assert metrics['post_harm'] == .25
    assert metrics['harmful_posts'] == 1
    assert metrics['n_posts'] == 2


def test_cp_uses_posts_and_binary_envelope_not_fractional_errors():
    records, rows = population()
    result = calibrate_posts(rows, records)
    assert len(result['curve']) == 101
    assert result['curve'][0]['n_posts'] == 2
    assert result['threshold'] is None
    assert result['upper'] == 0
    assert result['fallback'] == 'deterministic_all_off'


def test_maximum_coverage_and_smallest_threshold_tie():
    records = [{'id': str(index), 'tokens': [str(index)], 'images': ['image'], 'split': 'train'}
               for index in range(100)]
    rows = [{'id': str(index), 'gold': [1], 'text_prediction': [0],
             'vision_prediction': [1], 'gate': [.5], 'eligible': [True]}
            for index in range(100)]
    result = calibrate_posts(rows, records)
    assert result['threshold'] == 0
    assert result['detail_coverage'] == 1
    assert result['upper'] < .10
    assert result['post_harm'] == 0


def test_post_population_rejects_duplicate_missing_and_truncated_groups():
    records, rows = population()
    validate_population(rows, records, [row['id'] for row in rows])
    with pytest.raises(ValueError, match='IDs'):
        validate_population(rows + rows[:1], records, [row['id'] for row in rows])
    with pytest.raises(ValueError, match='missing relation'):
        validate_population(rows[:1], records, ['0'])
    rows[0]['gate'] = [math.nan]
    with pytest.raises(ValueError, match='finite'):
        validate_population(rows, records, [row['id'] for row in rows])


def test_no_image_and_alloff_preserve_text_predictions():
    records, rows = population()
    rows[0]['eligible'] = [False]
    output = routed_rows(rows, 0.)
    assert output[0]['prediction'] == rows[0]['text_prediction']
    assert post_metrics(routed_rows(rows, None), records)['post_harm'] == 0
