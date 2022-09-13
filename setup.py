from setuptools import find_packages, setup
from src import __version__

setup(
    name="nlp_sa",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    version=__version__,
    description="",
    author=""
)