from repro.p24.protocol import decoded_spans, split_records, task_f1
import torch


def test_calibration_groups_duplicate_posts_and_pilot_never_accesses_test():
    records = [{'id': str(index), 'tokens': ['same', str(index // 2)], 'images': ['image'], 'split': 'train'}
               for index in range(100)]
    records += [{'id': 'heldout', 'tokens': ['test'], 'images': [], 'split': 'test'}]
    result = split_records(records, 42)
    assert not {tuple(record['tokens']) for record in result['fit']} & {tuple(record['tokens']) for record in result['calibration']}
    assert split_records(records, 42, pilot=True)['test'] == []


def test_relation_f1_excludes_true_negative():
    rows = [{'prediction': [0], 'gold': [0]}, {'prediction': [1], 'gold': [1]},
            {'prediction': [2], 'gold': [1]}]
    assert task_f1(rows, 'mre') == .5


def test_span_decoder_stable_ties_and_type_exactness():
    probabilities = torch.tensor([[.1, .9], [.1, .9], [.8, .2]])
    spans = torch.tensor([[0, 2], [1, 3], [3, 4]])
    assert decoded_spans(probabilities, spans) == {(0, 2, 1)}
