import pytest
import spudtr


def test_get_ver():
    assert spudtr.__version__ == spudtr.get_ver()


@pytest.mark.xfail()
def test_get_ver_xfail():
    spudtr.__version__ = "bad_version"
    spudtr.get_ver()


@pytest.mark.parametrize(
    "_fname,_ftype",
    [
        ("sub000p3.ms100.epochs.feather", "feather"),
        pytest.param(
            "sub000p3.ms100.epochs.h5",
            "h5",
            marks=pytest.mark.xfail(
                strict=True, reason=NotImplementedError
            )
        )
    ]
)
@pytest.mark.parametrize("_url", [spudtr.DATA_URL, spudtr.DATA_URL[:-1]])
def test_get_demo_df(_fname, _ftype, _url):
    _ = spudtr.get_demo_df(_fname, _ftype, _url)
