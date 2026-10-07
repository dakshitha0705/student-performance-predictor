import pandas as pd

from src.monitor import make_shifted, shift_score, summarise_batch


def test_shift_score_formula():
    assert shift_score(11.0, 10.0, 4.0) == 0.25
    # tiny std is floored at 1 so the score cannot explode
    assert shift_score(11.0, 10.0, 0.2) == 1.0


def test_trigger_threshold_is_strictly_greater_than_half():
    reference = pd.DataFrame({"G1": [10] * 4, "G2": [10, 10, 10, 10], "studytime": [2] * 4})
    same = summarise_batch("same", reference, ref_mean=10.0, ref_std=4.0, synthetic=False)
    assert same["trigger"] == "no_shift"
    moved = reference.assign(G2=[8, 8, 8, 8])  # mean 8 -> score 0.5, NOT greater than 0.5
    assert summarise_batch("edge", moved, 10.0, 4.0, False)["trigger"] == "no_shift"
    beyond = reference.assign(G2=[7, 7, 7, 7])  # score 0.75
    assert summarise_batch("far", beyond, 10.0, 4.0, False)["trigger"] == "input_shift"


def test_shifted_batch_reduces_g2_by_five_and_clips_at_zero():
    batch = pd.DataFrame({"G1": [1, 2, 3], "G2": [3, 5, 12], "studytime": [1, 2, 3]})
    shifted = make_shifted(batch)
    assert shifted["G2"].tolist() == [0, 0, 7]
    assert batch["G2"].tolist() == [3, 5, 12]  # original untouched
