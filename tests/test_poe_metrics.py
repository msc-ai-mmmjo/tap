from olmo_tap.inference.poe import _validity_radius


def test_all_votes_against_target_needs_5_flips():
    # 8 heads all vote token 0; target = 1 has 0 votes
    # Need to flip 5 to get 5 vs 3 (strictly greater)
    assert _validity_radius([0] * 8, 1) == 5


def test_target_already_winning_returns_zero():
    # 5 for token 0 (target), 2 for 1, 1 for 2 → target already wins
    assert _validity_radius([0, 0, 0, 0, 0, 1, 1, 2], 0) == 0


def test_tied_competitors_exact():
    # 4 for token 0, 4 for token 1, target = 2 (0 votes)
    # Algorithm 1 from TPA paper gives 2 (wrong); greedy gives 4 (correct)
    assert _validity_radius([0, 0, 0, 0, 1, 1, 1, 1], 2) == 4


def test_multiple_competitors():
    # 6 for token 0, 2 for token 1, target = 2 (0 votes)
    assert _validity_radius([0, 0, 0, 0, 0, 0, 1, 1], 2) == 4


def test_target_has_some_votes():
    # 4 for token 0, 2 for token 1, 2 for target (= token 2)
    # need 2 flips: after 2, target has 4 vs max-opponent 2
    assert _validity_radius([0, 0, 0, 0, 1, 1, 2, 2], 2) == 2
