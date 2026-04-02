from setuptools import setup, find_packages

# Packages inside hyena-dna/src/ mapped via package_dir
# Must be listed explicitly because directory name contains a dash
HYENADNA_PACKAGES = [
    "src",
    "src.callbacks",
    "src.dataloaders",
    "src.dataloaders.datasets",
    "src.dataloaders.utils",
    "src.models",
    "src.models.nn",
    "src.models.sequence",
    "src.ops",
    "src.tasks",
    "src.utils",
    "src.utils.optim",
]

setup(
    name="euktect",
    version="1.0.0",
    description="Eukaryotic genome assessment tool using HyenaDNA (CPU-only conda package)",
    packages=find_packages(exclude=["docs", "tests", "hyena-dna"]) + HYENADNA_PACKAGES,
    package_dir={
        "src": "hyena-dna/src",
    },
    py_modules=["predict", "refine"],
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
        "pytorch-lightning>=1.8,<1.9",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
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
            "euktect-predict=euktect.cli:predict_main",
            "euktect-refine=euktect.cli:refine_main",
        ],
    },
)
