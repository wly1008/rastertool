# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 15:59:02 2026

@author: wly
"""

# tests/test_focal_basic.py
import numpy as np
import pytest

from rastertool.core.focaltools import (
    focal_mean,
    focal_sum,
    focal_min,
    focal_max,
    focal_var,
    focal_std,
    focal_count,
    focal_median,
    focal_perc,
    create_kernel,
)

@pytest.fixture
def test_raster():
    """
    5×5 测试栅格，中心为 NoData
    """
    data = np.array([
        [ 1,  2,  3,  4,  5],
        [ 6,  7, 8, 19, 20],
        [21, 22, 23, 24, 25],
    ], dtype=float)

    valid_mask = ~np.isnan(data)
    values = np.nan_to_num(data, nan=0.0)

    return values, valid_mask


@pytest.fixture
def kernel():
    # radius = 1 → 3×3
    return create_kernel(3, deleted=True)


@pytest.fixture
def cache():
    return {}


def test_focal_count_center(test_raster, kernel, cache):
    values, mask = test_raster

    result = focal_count(values, mask, kernel, mode="reflect", cval=0, cache=cache)

    # 中心像元 (2,2)
    assert result[2, 2] == 8


def test_focal_sum_center(test_raster, kernel, cache):
    values, mask = test_raster

    result = focal_sum(values, mask, kernel, mode="reflect", cval=0, cache=cache)

    assert result[2, 2] == 104


def test_focal_mean_center(test_raster, kernel, cache):
    values, mask = test_raster

    result = focal_mean(values, mask, kernel, mode="reflect", cval=0, cache=cache)

    assert result[2, 2] == pytest.approx(13.0)


def test_focal_min_max_center(test_raster, kernel, cache):
    values, mask = test_raster

    rmin = focal_min(values, mask, kernel, mode="reflect", cval=0, cache=cache)
    rmax = focal_max(values, mask, kernel, mode="reflect", cval=0, cache=cache)

    assert rmin[2, 2] == 7
    assert rmax[2, 2] == 19


def test_focal_var_std_center(test_raster, kernel, cache):
    values, mask = test_raster

    var = focal_var(values, mask, kernel, mode="reflect", cval=0, cache=cache)
    std = focal_std(values, mask, kernel, mode="reflect", cval=0, cache=cache)

    assert var[2, 2] == pytest.approx(18.5)
    assert std[2, 2] == pytest.approx(np.sqrt(18.5))


def test_focal_median_center(test_raster, kernel, cache):
    values, mask = test_raster

    median = focal_median(values, mask, kernel, mode="reflect", cval=0, cache=cache)

    assert median[2, 2] == pytest.approx(13.0)


def test_focal_percentile_50_center(test_raster, kernel, cache):
    values, mask = test_raster

    perc = focal_perc(
        values, mask, kernel,
        mode="reflect", cval=0,
        cache=cache, q=50
    )

    assert perc[2, 2] == pytest.approx(13.0)




