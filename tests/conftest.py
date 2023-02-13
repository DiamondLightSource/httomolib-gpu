# Defines common fixtures and makes them available to all tests

import os

import cupy as cp
import numpy as np
import pytest

CUR_DIR = os.path.abspath(os.path.dirname(__file__))


def pytest_addoption(parser):
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="run performance tests only",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "perf: mark test as performance test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--performance"):
        skip_other = pytest.mark.skip(reason="not a performance test")
        for item in items:
            if "perf" not in item.keywords:
                item.add_marker(skip_other)
    else:
        skip_perf = pytest.mark.skip(reason="performance test - use '--performance' to run")
        for item in items:
            if "perf" in item.keywords:
                item.add_marker(skip_perf)


@pytest.fixture(scope="session")
def test_data_path():
    return os.path.join(CUR_DIR, "test_data")


@pytest.fixture(scope="session")
def distortion_correction_path(test_data_path):
    return os.path.join(test_data_path, "distortion-correction")


# only load from disk once per session, and we use np.copy for the elements,
# to ensure data in this loaded file stays as originally loaded
@pytest.fixture(scope="session")
def data_file(test_data_path):
    in_file = os.path.join(test_data_path, "tomo_standard.npz")
    # keys: data, flats, darks, angles, angles_total, detector_y, detector_x
    return np.load(in_file)


@pytest.fixture
def ensure_clean_memory():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    yield None
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()



@pytest.fixture
def host_data(data_file):
    return np.copy(data_file["data"])


@pytest.fixture
def data(host_data, ensure_clean_memory):
    return cp.asarray(host_data)


@pytest.fixture
def host_flats(data_file):
    return np.copy(data_file["flats"])


@pytest.fixture
def flats(host_flats, ensure_clean_memory):
    return cp.asarray(host_flats)


@pytest.fixture
def host_darks(
    data_file,
):
    return np.copy(data_file["darks"])


@pytest.fixture
def darks(host_darks, ensure_clean_memory):
    return cp.asarray(host_darks)


@pytest.fixture
def host_angles(data_file):
    return np.copy(data_file["angles"])


@pytest.fixture
def angles(host_angles, ensure_clean_memory):
    return cp.asarray(host_angles)


@pytest.fixture
def host_angles_total(data_file):
    return np.copy(data_file["angles_total"])


@pytest.fixture
def angles_total(host_angles_total, ensure_clean_memory):
    return cp.asarray(host_angles_total)


@pytest.fixture
def host_detector_y(data_file):
    return np.copy(data_file["detector_y"])


@pytest.fixture
def detector_y(host_detector_y, ensure_clean_memory):
    return cp.asarray(host_detector_y)


@pytest.fixture
def host_detector_x(data_file):
    return np.copy(data_file["detector_x"])


@pytest.fixture
def detector_x(host_detector_x, ensure_clean_memory):
    return cp.asarray(host_detector_x)
