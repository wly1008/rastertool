# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 12:13:43 2026

@author: wly
"""








class RastertoolError(Exception):
    """Root exception class"""


class MergeError(RastertoolError):
    """Raised when rasters cannot be merged."""
    



class WindowError(RastertoolError):
    """Raised when errors occur during window operations"""