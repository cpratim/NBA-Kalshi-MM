from setuptools import setup, find_packages

setup(
    name="nba_kalshi_mm",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "nba-api>=1.2.1",
        "tqdm>=4.66.0",
        "requests>=2.31.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scipy>=1.10.0",
        "pbpstats>=1.1.0",
        "torch>=2.0.0",
    ],
    author="Pratim Chowdhary",
    author_email="cpratim18@gmail.com",
    description="Custom library for NBA Kalshi MM project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)
