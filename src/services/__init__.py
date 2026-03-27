"""
ForesightX Services Module
===========================
Centralized services for the ForesightX project including logging and other utilities.
"""

from .logger import get_logger, log_function_call

__all__ = ['get_logger', 'log_function_call']
