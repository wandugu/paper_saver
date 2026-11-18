import json

import pytest

from repro.p24.checkpoint_selection import checkpoint_plan, policy_disagreement


def test_phase1_only_best_is_blocked_without_substituting(tmp_path):
    (tmp_path / 'best.pt').touch()
    plan = checkpoint_plan(tmp_path)
    assert not plan['complete']
    assert plan['available_checkpoints'] == ['best.pt']


def test_selection_uses_dev_scores_and_earliest_ties(tmp_path):
    epochs = [{'epoch': epoch, 'checkpoint': f'epoch_{epoch}.pt',
               'soft_dev_f1': soft, 'hard_dev_f1': hard}
              for epoch, soft, hard in [(0, .8, .5), (1, .8, .9), (2, .7, .6)]]
    for row in epochs:
        (tmp_path / row['checkpoint']).touch()
    (tmp_path / 'epochs.json').write_text(json.dumps({'epochs': epochs}))
    selected = checkpoint_plan(tmp_path)['selected']
    assert selected['soft_selected']['epoch'] == 0
    assert selected['hard_selected']['epoch'] == 1
    assert selected['last_epoch']['epoch'] == 2


def test_missing_checkpoint_does_not_use_a_different_epoch(tmp_path):
    (tmp_path / 'epochs.json').write_text(json.dumps({'epochs': [
        {'epoch': 0, 'checkpoint': 'missing.pt', 'soft_dev_f1': .9, 'hard_dev_f1': .9}]}))
    assert not checkpoint_plan(tmp_path)['complete']


def test_disagreement_joins_by_identity():
    rows = [{'id': 'a', 'gold': [1], 'active': [True]},
            {'id': 'b', 'gold': [0], 'active': [False]}]
    other = [dict(rows[1]), {**rows[0], 'active': [False]}]
    assert policy_disagreement(rows, other) == .5
    with pytest.raises(ValueError, match='identities'):
        policy_disagreement(rows, other[:1])
