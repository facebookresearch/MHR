from setuptools import find_packages, setup

setup(
    name="mhr",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "pymomentum-gpu>=0.1.84",
        "scikit-learn>=1.7.2,<2",
        "smplx>=0.1.28,<0.2",
        "torch",
        "tqdm>=4.67.1,<5",
        "trimesh>=4.8.3,<5",
    ],
)
