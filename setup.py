#!/usr/bin/env python3

from setuptools import setup, find_packages
import os
import sys
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ""

# Read the requirements file
def read_requirements(filename):
    """Read requirements from requirements.txt file."""
    requirements_path = Path(__file__).parent / filename
    if requirements_path.exists():
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Get version from config.json or set default
def get_version():
    """Extract version from config.json or use default."""
    try:
        import json
        config_path = Path(__file__).parent / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get('project', {}).get('version', '1.0.0')
    except:
        pass
    return '1.0.0'

# Define package data
package_data = {
    'aerospace_optimizer': [
        'data/*.csv',
        'data/*.json', 
        'config/*.json',
        'config/*.yaml',
        'examples/*.py',
        'tests/*.py',
        'visualization/templates/*.html',
        'docs/*.md'
    ]
}

# Define entry points for command-line scripts
entry_points = {
    'console_scripts': [
        'amf-sbo=src.cli:main',
        'aerospace-optimizer=src.cli:main',
        'amf-sbo-test=tests.run_all_tests:main',
        'amf-sbo-examples=examples.run_all_examples:main'
    ]
}

# Classifiers for PyPI
classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Software Development :: Libraries :: Python Modules'
]

# Keywords for PyPI
keywords = [
    'aerospace', 'optimization', 'multi-fidelity', 'simulation',
    'uncertainty quantification', 'robust design', 'aircraft',
    'spacecraft', 'genetic algorithm', 'particle swarm',
    'bayesian optimization', 'multi-objective'
]

setup(
    # Basic package information
    name='adaptive-multifidelity-aerospace-optimization',
    version=get_version(),
    author='Aerospace Optimization Research Team',
    author_email='contact@aerospace-optimization.org',
    maintainer='Aerospace Optimization Research Team',
    maintainer_email='contact@aerospace-optimization.org',
    
    # Package description
    description='Adaptive Multi-Fidelity Simulation-Based Optimization for Aerospace Systems',
    long_description=long_description,
    long_description_content_type='text/markdown',
    
    # URLs
    url='https://github.com/aerospace-optimization/adaptive-multifidelity',
    project_urls={
        'Bug Reports': 'https://github.com/aerospace-optimization/adaptive-multifidelity/issues',
        'Source': 'https://github.com/aerospace-optimization/adaptive-multifidelity',
        'Documentation': 'https://aerospace-optimization.github.io/adaptive-multifidelity',
        'Funding': 'https://github.com/sponsors/aerospace-optimization'
    },
    
    # Package discovery and organization
    packages=find_packages(where='.', include=['src*', 'examples*', 'tests*']),
    package_dir={'': '.'},
    package_data=package_data,
    include_package_data=True,
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Dependencies
    install_requires=read_requirements('requirements.txt'),
    extras_require={
        'dev': read_requirements('requirements-dev.txt') if Path('requirements-dev.txt').exists() else [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'black>=21.0.0',
            'flake8>=4.0.0',
            'mypy>=0.910',
            'pre-commit>=2.15.0',
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0'
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0',
            'myst-parser>=0.15.0',
            'nbsphinx>=0.8.0'
        ],
        'viz': [
            'plotly>=5.0.0',
            'bokeh>=2.4.0',
            'dash>=2.0.0',
            'streamlit>=1.0.0'
        ],
        'hpc': [
            'mpi4py>=3.1.0',
            'dask>=2021.9.0',
            'ray>=1.6.0'
        ]
    },
    
    # Entry points for command-line tools
    entry_points=entry_points,
    
    # Metadata
    classifiers=classifiers,
    keywords=keywords,
    license='MIT',
    
    # Data files
    data_files=[
        ('config', ['config.json']),
        ('examples', [
            'examples/aircraft_optimization_example.py',
            'examples/spacecraft_optimization_example.py',
            'examples/run_all_examples.py'
        ]),
        ('tests', [
            'tests/test_simulation_framework.py',
            'tests/test_optimization_algorithms.py',
            'tests/run_all_tests.py'
        ])
    ],
    
    # Archive options
    zip_safe=False,
    
    # Additional metadata
    platforms=['any'],
    
    # Test suite
    test_suite='tests',
    tests_require=[
        'pytest>=6.0.0',
        'pytest-cov>=2.10.0',
        'hypothesis>=6.0.0'
    ],
    
    # Command options
    options={
        'build_scripts': {
            'executable': '/usr/bin/env python3'
        }
    }
)

# Post-installation message
if __name__ == '__main__':
    print("\n" + "="*80)
    print("ADAPTIVE MULTI-FIDELITY AEROSPACE OPTIMIZATION INSTALLATION")
    print("="*80)
    print("✓ Package installed successfully!")
    print("\nQuick start commands:")
    print("  amf-sbo-test          # Run all tests")
    print("  amf-sbo-examples      # Run example optimizations")
    print("  aerospace-optimizer   # Launch CLI interface")
    print("\nFor documentation and examples, visit:")
    print("  https://aerospace-optimization.github.io/adaptive-multifidelity")
    print("="*80)