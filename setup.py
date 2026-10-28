from setuptools import setup, find_packages

setup(
    name="end2reg",
    version="1.0.0",
    description="End2Reg",
    author="Lorenzo Pettinari",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "numpy",
        "easydict",
        "einops",
        "open3d>=0.18.0",
        "scipy",
        "scikit-learn",
        "tqdm",
        "pandas",
        "openpyxl",
        "coloredlogs",
        "tensorboard",
        "matplotlib",
    ],
)
