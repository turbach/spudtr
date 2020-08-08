from pathlib import Path
import pytest
import spudtr


def test_get_ver():
    assert spudtr.__version__ == spudtr.get_ver()


@pytest.mark.xfail()
def test_get_ver_xfail():
    spudtr.__version__ = "bad_version"
    spudtr.get_ver()


@pytest.mark.parametrize("_url", [spudtr.DATA_URL, spudtr.DATA_URL[:-1]])
def test_get_demo_df(_url):
    test_f = "sub000p3.ms100.epochs.feather"
    if (spudtr.DATA_DIR / test_f).exists():
        (spudtr.DATA_DIR / test_f).unlink()

    _ = spudtr.get_demo_df(test_f, _url)  # download and cache
    _ = spudtr.get_demo_df(test_f, _url)  # read cached
