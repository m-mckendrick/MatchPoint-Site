import pandas as pd
from src.feature_engineering.match_results import compute_match_result


def test_whitebox_games_parsing_with_special_rule():
    # 7-6 should be counted as 6-6
    score = "7-6(4)"
    tb = "7-4"
    p1_win = 1

    (
        p1_sets, p2_sets,
        p1_games, p2_games,
        p1_tb, p2_tb,
        p1_ss, p2_ss
    ) = compute_match_result(score, tb, p1_win)

    assert p1_sets == 1
    assert p2_sets == 0
    assert p1_games == 6
    assert p2_games == 6
    assert p1_tb == 1
    assert p2_tb == 0


def test_whitebox_multiple_sets_with_tiebreak_alignment():
    score = "7-6(4) 6-7(8) 6-2"
    tb = "7-4 10-8 none"
    p1_win = 1

    (
        p1_sets, p2_sets,
        p1_games, p2_games,
        p1_tb, p2_tb,
        p1_ss, p2_ss
    ) = compute_match_result(score, tb, p1_win)

    assert p1_sets == 2
    assert p2_sets == 1
    assert p1_tb == 1
    assert p2_tb == 1
