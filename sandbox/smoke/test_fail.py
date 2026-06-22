"""Smoke fixture: a trivially FAILING test set.

Run inside activelearning-sandbox:latest to prove the image reports failure
(non-zero exit code) for broken generated code — the sandbox must distinguish
pass from fail, not merely "ran".
"""


def test_obviously_false():
    assert 1 + 1 == 3
