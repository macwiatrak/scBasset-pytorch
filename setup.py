from setuptools import find_packages, setup

setup(
    name="scbasset",
    packages=find_packages(),
    package_data={
        "scBasset": ["py.typed", "**/*.json", "**/*.yaml"],
    },
    install_requires=[
        "numpy~=1.24.1",
        "torch~=1.13.1",
    ],
    extras_require={
        "testing": [
            "pytest~=7.2.0",
        ]
    },
)
