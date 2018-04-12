import pytest


@pytest.fixture
def medembed():
    import main
    main.get_arguments()

def test_func(medembed):
    assert medembed == 0
