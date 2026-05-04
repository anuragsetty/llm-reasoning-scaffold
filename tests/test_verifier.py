from scaffold.verifier import is_valid_equation


def test_valid_equation():
    assert is_valid_equation("N_00 + N_01")


def test_invalid_equation():
    assert not is_valid_equation("")
    assert not is_valid_equation("not an equation@@@")
