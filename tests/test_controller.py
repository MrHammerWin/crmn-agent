import math

import numpy as np
import pytest

from crmn_agent.controller import Prior, cycle_error, entropy, features, prediction_error, responsibilities
from crmn_agent.schemas import Anchor, Audit, Fact, Reconstruction, Verdict


def anchor(key='carry', value='red ball'):
    return Anchor(key=key, value=value, evidence='You carry a red ball')


def test_cycle_preservation():
    assert cycle_error([anchor()], Reconstruction(facts=[Fact(key='carry', value='red ball')])) == 0


def test_missing_known_fact_is_penalized():
    assert cycle_error([anchor()], Reconstruction(facts=[])) == 1


def test_unknown_unseen_fact_is_not_false():
    assert cycle_error([], Reconstruction(facts=[Fact(key='door', value='open')])) == 0


def test_cycle_case_and_whitespace():
    assert cycle_error([anchor()], Reconstruction(facts=[Fact(key=' CARRY ', value='Red Ball')])) == 0


@pytest.mark.parametrize('status,evidence,expected_count,expected_error', [
    ('unknown', '', 0, 0), ('contradicted', 'not in view', 0, 0),
    ('contradicted', 'You carry nothing', 1, 1),
    ('supported', 'You carry nothing', 1, 0),
])
def test_predictions_require_observable_literal_evidence(status, evidence, expected_count, expected_error):
    audit = Audit(verdicts=[Verdict(prediction='carrying', status=status, evidence=evidence)])
    assert prediction_error(audit, 'You carry nothing', ['carrying']) == (expected_error, expected_count)


def test_unexecuted_counterfactual_is_not_scored():
    verdict = Verdict(prediction='other action worked', status='contradicted', evidence='wall')
    assert prediction_error(Audit(verdicts=[verdict]), 'wall', ['actual prediction']) == (0, 0)


def test_duplicate_verdicts_do_not_multiply_evidence():
    v = Verdict(prediction='p', status='contradicted', evidence='wall')
    assert prediction_error(Audit(verdicts=[v, v]), 'wall', ['p']) == (1, 1)


def test_error_gates_unreliable_representation():
    weights = responsibilities([0, 1], [0.5, 0.5], [0.5, 0.5])
    assert weights[0] > weights[1]
    assert sum(weights) == pytest.approx(1)


def test_utility_can_break_consistent_but_useless_tie():
    assert responsibilities([0, 0], [0.5, 0.5], [0, 1])[1] > 0.5


def test_numerical_stability():
    w = responsibilities([1e9, 1e9 + 1], [0.5, 0.5], [0, 0])
    assert all(math.isfinite(v) for v in w)
    assert entropy(w) >= 0


@pytest.mark.parametrize('errors,priors,values', [([], [], []), ([0], [0.5, 0.5], [0]),
                                                ([float('nan')], [1], [0]), ([0], [-1], [0])])
def test_bad_gate_inputs_fail(errors, priors, values):
    with pytest.raises(ValueError):
        responsibilities(errors, priors, values)


def test_prior_cannot_train_on_test_data():
    with pytest.raises(ValueError):
        Prior().fit([{'split': 'test', 'features': [1] * 6, 'responsibilities': [1, 0]}])


def test_prior_learns_and_roundtrips(tmp_path):
    prior = Prior()
    x = features('wall', 1, [])
    prior.fit([{'split': 'train', 'features': x.tolist(), 'responsibilities': [0.95, 0.05]}])
    assert prior.predict(x)[0] > 0.8
    prior.save(tmp_path / 'prior.json')
    assert Prior.load(tmp_path / 'prior.json').predict(x) == pytest.approx(prior.predict(x))


def test_bad_prior_shape(tmp_path):
    (tmp_path / 'bad.json').write_text('{"weights": [[0]], "training_rows": 1}')
    with pytest.raises(ValueError):
        Prior.load(tmp_path / 'bad.json')


def test_uniform_prior_is_explicitly_untrained():
    prior = Prior()
    assert prior.predict(np.ones(6)) == [0.5, 0.5]
    assert prior.training_rows == 0

