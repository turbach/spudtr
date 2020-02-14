import pytest
import spudtr


def test_get_ver():
    assert spudtr.__version__ == spudtr.get_ver()


@pytest.mark.xfail()
def test_get_ver_xfail():
    spudtr.__version__ = "bad_version"
    spudtr.get_ver()
