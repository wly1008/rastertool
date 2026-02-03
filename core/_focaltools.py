# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 15:53:13 2026

@author: wly
"""

import numpy as np
from numba import njit, prange


@njit(inline='always')
def _idx_nearest(i, n):
    if i < 0:
        return 0
    if i >= n:
        return n - 1
    return i


@njit(inline='always')
def _idx_wrap(i, n):
    return i % n


@njit(inline='always')
def _idx_reflect(i, n):
    if n == 1:
        return 0
    while i < 0 or i >= n:
        if i < 0:
            i = -i
        else:
            i = 2 * n - 2 - i
    return i


@njit(inline='always')
def _idx_mirror(i, n):
    if n == 1:
        return 0
    while i < 0 or i >= n:
        if i < 0:
            i = -i - 1
        else:
            i = 2 * n - 1 - i
    return i

# @njit(inline='always')
# def _idx_dispatch(i, n, mode):
#     if mode == 0:
#         return _idx_nearest(i, n)
#     elif mode == 1:
#         return _idx_reflect(i, n)
#     elif mode == 2:
#         return _idx_mirror(i, n)
#     else:
#         return _idx_wrap(i, n)
    
@njit(inline='always', fastmath=True)
def _idx_dispatch(i, n, mode):
    if mode == 0:      # nearest
        if i < 0:
            return 0
        if i >= n:
            return n - 1
        return i

    elif mode == 1:    # reflect
        if n == 1:
            return 0
        period = 2 * n - 2
        i = i % period
        return i if i < n else period - i

    elif mode == 2:    # mirror
        if n == 1:
            return 0
        period = 2 * n
        i = i % period
        return i if i < n else period - 1 - i

    else:              # wrap
        return i % n



@njit(parallel=True)
def percentile_filter_array_q(
    input,
    percentile,
    footprint,
    mode=1,
    cval=0.0,
):
    """
    对二维数组应用百分位数滤波（percentile filter）。
    
    该函数在给定的邻域（由 footprint 指定）内，计算输入数组每个像素点
    对应邻域像素值的指定百分位数，并生成输出数组。
    使用 Numba JIT 编译并行加速，支持多种边界处理模式。
    
    参数
    ----------
    input : ndarray, shape (H, W)
        输入的二维数组。
    
    percentile : float 或 ndarray, shape (H, W)
        百分位数，取值范围为 [0, 100]。
        - 若为标量，则对所有像素使用同一个百分位数；
        - 若为二维数组，则每个像素位置使用对应的百分位数。
    
    footprint : ndarray of bool, shape (fh, fw)
        邻域掩膜，非零（True）元素表示参与百分位数计算的邻域位置。
        邻域中心位于 footprint 的中心位置。
    
    mode : int, 可选
        边界处理模式，默认值为 1。
    
        可选值说明：
        - 0 : nearest
              使用最近的边界值填充越界索引
        - 1 : reflect
              反射边界（不重复边界点）
        - 2 : mirror
              镜像边界（重复边界点）
        - 3 : wrap
              周期性边界
        - 4 : constant
              使用常数 cval 填充越界区域
    
    cval : scalar, 可选
        当 mode == 4（constant）时使用的填充值，默认值为 0.0。
    
    返回
    -------
    out : ndarray, shape (H, W)
        百分位数滤波后的输出数组，形状与 input 相同。
    
    说明
    ----
    - 百分位数的计算基于离散排序结果，不进行插值，
      等价于取排序后索引为 floor(q/100 * (N-1)) 的元素。
    - 内部使用 np.partition 实现部分排序，以提高性能。
    - 该函数假设 input 为二维数组。
    """
    h, w = input.shape
    fh, fw = footprint.shape
    rh, rw = fh // 2, fw // 2

    out = np.empty_like(input)
    max_n = fh * fw
    q_is_scalar = np.ndim(percentile) == 0

    for i in prange(h):
        buf = np.empty(max_n, input.dtype)  # 每个prange一个buf
        for j in range(w):
            n = 0

            if mode == 4:  # constant 特殊处理
                for di in range(-rh, rh + 1):
                    for dj in range(-rw, rw + 1):
                        if not footprint[di + rh, dj + rw]:
                            continue
                        ii = i + di
                        jj = j + dj
                        if ii < 0 or ii >= h or jj < 0 or jj >= w:
                            buf[n] = cval
                        else:
                            buf[n] = input[ii, jj]
                        n += 1
            else:
                for di in range(-rh, rh + 1):
                    for dj in range(-rw, rw + 1):
                        if not footprint[di + rh, dj + rw]:
                            continue
                        ii = _idx_dispatch(i + di, h, mode)
                        jj = _idx_dispatch(j + dj, w, mode)
                        buf[n] = input[ii, jj]
                        n += 1

            q = percentile if q_is_scalar else percentile[i, j]
            if q <= 0.0:
                k = 0
            elif q >= 100.0:
                k = n - 1
            else:
                k = int((q / 100.0) * (n - 1))

            tmp = buf[:n]
            np.partition(tmp, k)
            out[i, j] = tmp[k]

    return out















