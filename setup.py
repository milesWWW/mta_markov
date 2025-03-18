from setuptools import setup, find_packages

setup(
    name="mta_markov",
    version="0.1.0",
    author="miles.wang",
    author_email="miles.wang22@gmail.com",
    description="A Markov Chain based Multi-Touch Attribution (MTA) model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/milesWWW/mta_markov",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
