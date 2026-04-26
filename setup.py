from setuptools import setup, find_packages

setup(
    name="hypnos",
    version="0.1.0",
    description="Continuous latent reasoning with autonomous dream consolidation",
    author="Rajat Malik",
    author_email="",
    url="https://github.com/Rajat25022005/hypnos",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "pyyaml>=6.0",
        "einops>=0.7.0",
        "numpy>=1.26.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.12",
    ],
)
