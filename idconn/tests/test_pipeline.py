"""Tests for idconn.pipeline."""


def test_idconn_workflow_smoke():
    '''
    this is a docstring bc my tests kept failing and it was annoying
    '''
    from idconn.pipeline import idconn_workflow

    # Check that it's a function ¯\_(ツ)_/¯
    assert callable(idconn_workflow)
