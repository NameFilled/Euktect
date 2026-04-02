from setuptools import setup, find_packages

setup(
    name="euktect",
    version="1.0.0",
    description="Eukaryotic genome assessment tool using HyenaDNA (CPU-only conda package)",
    packages=find_packages(exclude=["docs", "tests"]),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "rich",
        "pytorch-lightning==1.8.6",
        "hydra-core",
        "omegaconf",
        "einops",
        "opt_einsum",
        "transformers==4.26.1",
        "timm",
        "pyfaidx",
        "polars",
        "loguru",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "euktect-predict=predict:main",
            "euktect-refine=refine:main",
        ],
    },
)
