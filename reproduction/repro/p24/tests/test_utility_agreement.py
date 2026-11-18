import pytest

from repro.p24.utility_agreement import agreement


def rows(targets):
    return [{'unit_id': str(index), 'target': target, 'eligible': True}
            for index, target in enumerate(targets)]


def test_perfect_and_chance_adjusted_agreement():
    assert agreement(rows([0, 1, 0, 1]), rows([0, 1, 0, 1]))['kappa'] == 1
    output = agreement(rows([0, 1, 0, 1]), rows([1, 1, 0, 0]))
    assert output['agreement'] == .5
    assert output['kappa'] == 0


def test_single_class_kappa_is_undefined_and_identity_mismatch_rejected():
    assert agreement(rows([1, 1]), rows([1, 1]))['kappa'] is None
    with pytest.raises(ValueError, match='exactly'):
        agreement(rows([0, 1]), rows([0]))
    with pytest.raises(ValueError, match='Duplicate'):
        agreement(rows([0]) * 2, rows([0]))
