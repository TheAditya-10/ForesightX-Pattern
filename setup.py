from setuptools import find_packages, setup


setup(
    name="foresightx-pattern",
    packages=find_packages(exclude=("tests", "docs")),
    version="1.0.0",
    description="Production-grade ML microservice for multi-stock pattern forecasting.",
    author="Aditya Pratap Singh Tomar",
    license="MIT",
)
