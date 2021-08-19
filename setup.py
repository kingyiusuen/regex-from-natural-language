from pathlib import Path

from setuptools import setup


BASE_DIR = Path(__file__).parent


with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]


dev_packages = [
    "black",
    "flake8",
    "isort",
    "mypy",
    "pre-commit",
]


setup(
    name="regex-from-natural-language",
    version="0.1",
    license="MIT",
    description="Generation of regular expressions from natural language.",
    author="King Yiu Suen",
    author_email="kingyiusuen@gmail.com",
    url="https://github.com/kingyiusuen/regex-from-natural-language/",
    keywords=[
        "machine-learning",
        "deep-learning",
        "artificial-intelligence",
        "latex",
        "neural-network",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
    install_requires=[required_packages],
    extras_require={
        "dev": dev_packages,
    },
)
