from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read version from version.py
version_ns = {}
with open(os.path.join(this_directory, 'tra_algorithm', 'version.py')) as f:
    exec(f.read(), version_ns)

setup(
    name="tra-algorithm",
    version=version_ns["__version__"],
    author="Dasari Ranga Eswar",
    author_email="rangaeswar890@gmail.com",
    description="Track/Rail Algorithm (TRA) - A novel machine learning algorithm for dynamic model selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eswaroy/tra_algorithm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.3.0",
        "joblib>=1.0.0",
        "networkx>=2.5",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    keywords="machine learning, algorithm, dynamic model selection, ensemble, classification, regression",
    project_urls={
        "Bug Reports": "https://github.com/eswaroy/tra_algorithm/issues",
        "Source": "https://github.com/eswaroy/tra_algorithm",
        "Documentation": "https://tra-algorithm.readthedocs.io/",
    },
    include_package_data=True,
    zip_safe=False,
)