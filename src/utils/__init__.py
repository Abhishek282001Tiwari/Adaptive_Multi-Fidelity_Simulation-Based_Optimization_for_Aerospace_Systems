"""
Utility modules for the Adaptive Multi-Fidelity Aerospace Optimization Framework

This package contains utility functions and classes that support the main
optimization framework, including data generation, file management, and
helper functions.

Author: Aerospace Optimization Research Team
Version: 1.0.0
Certification: NASA-STD-7009A & AIAA-2021-0123 Compliant
"""

from .local_data_generator import LocalDataGenerator

__all__ = [
    'LocalDataGenerator'
]

__version__ = '1.0.0'
__author__ = 'Aerospace Optimization Research Team'
__certification__ = 'AMFSO-2024-001'