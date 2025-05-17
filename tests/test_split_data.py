from data.split_data import split_data


def test_split_data():
    X = [[i] for i in range(10)]
    y = [0, 1] * 5

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.3, random_state=0)

    assert len(X_train) == 7
    assert len(X_test) == 3
    assert len(y_train) == 7
    assert len(y_test) == 3

    assert set(y_train + y_test) == set(y), "Mismatch in label values after split"
