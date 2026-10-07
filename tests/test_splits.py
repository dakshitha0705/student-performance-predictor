from src.split import make_partitions

from tests.conftest import synthetic_frame


def test_partitions_are_disjoint_and_complete():
    frame = synthetic_frame(395)  # same size as the mathematics file
    parts = make_partitions(frame)
    all_ids = [i for p in parts.values() for i in p["source_row_id"]]
    assert len(all_ids) == len(set(all_ids))          # disjoint
    assert set(all_ids) == set(frame["source_row_id"])  # complete


def test_same_seed_gives_identical_assignments():
    frame = synthetic_frame(395)
    first = make_partitions(frame)
    second = make_partitions(frame)
    for name in first:
        assert first[name]["source_row_id"].tolist() == second[name]["source_row_id"].tolist()


def test_sizes_are_about_60_15_15_10():
    frame = synthetic_frame(400)
    parts = make_partitions(frame)
    shares = {k: len(v) / 400 for k, v in parts.items()}
    assert abs(shares["train"] - 0.60) < 0.02
    assert abs(shares["validation"] - 0.15) < 0.02
    assert abs(shares["test"] - 0.15) < 0.02
    assert abs(shares["later"] - 0.10) < 0.02


def test_later_and_test_rows_never_in_train():
    parts = make_partitions(synthetic_frame(395))
    train_ids = set(parts["train"]["source_row_id"])
    assert not train_ids & set(parts["later"]["source_row_id"])
    assert not train_ids & set(parts["test"]["source_row_id"])
    assert not set(parts["validation"]["source_row_id"]) & set(parts["test"]["source_row_id"])
