"""Tests for idconn.pipeline."""


def test_idconn_workflow_smoke():
    from idconn.pipeline import idconn_workflow

    # Check that it's a function ¯\_(ツ)_/¯
    assert callable(idconn_workflow)
