from crmn_agent.grounding import observation_anchors


def test_all_anchors_are_real_substrings():
    text = 'You see a wall 2 steps forward.\nYou carry a blue key.'
    assert all(a.evidence in text and a.value == a.evidence for a in observation_anchors(text, 'blue key'))


def test_goal_relevant_observations_rank_first():
    anchors = observation_anchors('Wall ahead.\nBlue key on floor.', 'pick up the blue key', limit=1)
    assert anchors[0].value == 'Blue key on floor.'


def test_anchor_limit_and_long_text():
    text = ('object ' * 100) + '\n' + 'wall. ' * 100
    result = observation_anchors(text, 'object', limit=4)
    assert len(result) == 4
    assert all(len(a.value) <= 110 and a.evidence in text for a in result)


def test_no_observation_means_no_knowledge():
    assert observation_anchors('', 'the unseen key is blue') == []
