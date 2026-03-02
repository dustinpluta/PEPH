import pandas as pd

from peph.data.split import train_test_split_subject, apply_split


def test_subject_split_no_leakage() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3, 4, 4],
            "time": [1, 2, 3, 4, 5, 6, 7, 8],
            "event": [0, 1, 0, 0, 1, 0, 0, 0],
        }
    )

    split = train_test_split_subject(df, id_col="id", test_size=0.5, seed=123)
    train, test = apply_split(df, id_col="id", split=split)

    assert set(train["id"].unique()).isdisjoint(set(test["id"].unique()))
    assert len(train) + len(test) == len(df)