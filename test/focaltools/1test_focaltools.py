# -*- coding: utf-8 -*-
"""
focal_* 核心函数单元测试
不依赖 focal_tool / pytest
"""

import numpy as np

# ===== 修改为你的模块名 =====
from  rastertool.core.focaltools import (
    create_kernel,
    focal_count,
    focal_sum,
    focal_mean,
    focal_min,
    focal_max,
    focal_var,
    focal_std,
    focal_perc,
    focal_median,
)
# ============================


MODE = "reflect"
CVAL = 0.0


def assert_close(a, b, msg):
    if not np.isclose(a, b):
        raise AssertionError(f"{msg}: {a} != {b}")


def print_ok(name):
    print(f"✔ {name} passed")


# --------------------------------------------------
# 测试数据
# --------------------------------------------------

values = np.array(
    [[1, 2, 3],
     [4, 0, 6],
     [7, 8, 9]],
    dtype=float
)

# 中心像元无效
valid_mask = np.array(
    [[1, 1, 1],
     [1, 0, 1],
     [1, 1, 1]],
    dtype=bool
)

kernel = create_kernel(3, deleted=False)


# --------------------------------------------------
# focal_count
# --------------------------------------------------

def test_focal_count():
    cache = {}
    out = focal_count(values, valid_mask, kernel, mode=MODE, cval=CVAL, cache=cache)
    assert out[1, 1] == 8, "focal_count 错误"
    print_ok("focal_count")


# --------------------------------------------------
# focal_sum
# --------------------------------------------------

def test_focal_sum():
    cache = {}
    out = focal_sum(values, valid_mask, kernel, mode=MODE, cval=CVAL, cache=cache)

    expected = 1 + 2 + 3 + 4 + 6 + 7 + 8 + 9
    assert_close(out[1, 1], expected, "focal_sum 错误")
    print_ok("focal_sum")


# --------------------------------------------------
# focal_mean
# --------------------------------------------------

def test_focal_mean():
    cache = {}
    out = focal_mean(values, valid_mask, kernel, mode=MODE, cval=CVAL, cache=cache)

    expected = (1 + 2 + 3 + 4 + 6 + 7 + 8 + 9) / 8
    assert_close(out[1, 1], expected, "focal_mean 错误")
    print_ok("focal_mean")


# --------------------------------------------------
# focal_min / max
# --------------------------------------------------

def test_focal_min_max():
    cache = {}

    mn = focal_min(values, valid_mask, kernel, mode=MODE, cval=CVAL, cache=cache)
    mx = focal_max(values, valid_mask, kernel, mode=MODE, cval=CVAL, cache=cache)

    assert mn[1, 1] == 1, "focal_min 错误"
    assert mx[1, 1] == 9, "focal_max 错误"
    print_ok("focal_min / focal_max")


# --------------------------------------------------
# focal_var / std
# --------------------------------------------------

def test_focal_var_std():
    cache = {}

    var = focal_var(values, valid_mask, kernel, mode=MODE, cval=CVAL, cache=cache)
    std = focal_std(values, valid_mask, kernel, mode=MODE, cval=CVAL, cache=cache)

    vals = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=float)
    expected_var = vals.var()

    assert_close(var[1, 1], expected_var, "focal_var 错误")
    assert_close(std[1, 1], np.sqrt(expected_var), "focal_std 错误")
    print_ok("focal_var / focal_std")


# --------------------------------------------------
# focal_perc / median
# --------------------------------------------------

def test_focal_percentile():
    cache = {}

    p50 = focal_perc(
        values,
        valid_mask,
        kernel,
        mode=MODE,
        cval=CVAL,
        cache=cache,
        q=50,
    )

    med = focal_median(
        values,
        valid_mask,
        kernel,
        mode=MODE,
        cval=CVAL,
        cache={},
    )

    assert p50[1, 1] == 4, "focal_perc q=50 错误"
    assert med[1, 1] == 4, "focal_median 错误"
    print_ok("focal_perc / focal_median")


# --------------------------------------------------
# deleted=True
# --------------------------------------------------

def test_deleted_center():
    kernel_del = create_kernel(3, deleted=True)
    cache = {}

    out = focal_mean(values, valid_mask, kernel_del, mode=MODE, cval=CVAL, cache=cache)

    expected = (1 + 2 + 3 + 4 + 6 + 7 + 8 + 9) / 8
    assert_close(out[1, 1], expected, "deleted=True 错误")
    print_ok("deleted=True")


# --------------------------------------------------
# cache 复用
# --------------------------------------------------

def test_cache_reuse():
    cache = {}

    focal_mean(values, valid_mask, kernel, mode=MODE, cval=CVAL, cache=cache)
    focal_std(values, valid_mask, kernel, mode=MODE, cval=CVAL, cache=cache)

    for v in cache.values():
        assert np.isfinite(v).all(), "cache 中存在非法值"

    print_ok("cache reuse")


# --------------------------------------------------
# 主入口
# --------------------------------------------------

def run_all_tests():
    test_focal_count()
    test_focal_sum()
    test_focal_mean()
    test_focal_min_max()
    test_focal_var_std()
    test_focal_percentile()
    test_deleted_center()
    test_cache_reuse()

    print("\n✅ 所有 focal_* 核心函数测试通过")


if __name__ == "__main__":
    run_all_tests()
