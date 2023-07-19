"""
trainy is an observability tool for profiling PyTorch training on demand.
"""

import io
import os
import re

import setuptools

ROOT_DIR = os.path.dirname(__file__)


def find_version(*filepath):
    with open(os.path.join(ROOT_DIR, *filepath)) as fp:
        version_match = re.search(
            r'^__version__ = [\'"]([^\'"]*)[\'"]', fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


def parse_readme(readme: str) -> str:
    """Parse the README.md file to be pypi compatible."""
    # Replace the footnotes.
    readme = readme.replace("<!-- Footnote -->", "#")
    footnote_re = re.compile(r"\[\^([0-9]+)\]")
    readme = footnote_re.sub(r"<sup>[\1]</sup>", readme)

    # Remove the dark mode switcher
    mode_re = re.compile(
        r"<picture>[\n ]*<source media=.*>[\n ]*<img(.*)>[\n ]*</picture>", re.MULTILINE
    )
    readme = mode_re.sub(r"<img\1>", readme)
    return readme


install_requires = [
    "wheel",
    "ray[default]==2.4.0",
    "torch>=2.0.0",
    "gpustat==1.0",
    "posthog",
]

long_description = ""
readme_filepath = "README.md"
# When sky/backends/wheel_utils.py builds wheels, it will not contain the
# README.  Skip the description for that case.
if os.path.exists(readme_filepath):
    long_description = io.open(readme_filepath, "r", encoding="utf-8").read()
    long_description = parse_readme(long_description)

setuptools.setup(
    name="trainy",
    version=find_version("trainy", "__init__.py"),
    packages=setuptools.find_packages(),
    author="Trainy Team",
    license="Apache 2.0",
    readme="README.md",
    description="Trainy: An observability tool for profiling PyTorch training on demand",
    long_description=long_description,
    long_description_content_type="text/markdown",
    setup_requires=["wheel"],
    requires_python=">=3.7",
    install_requires=install_requires,
    entry_points={
        "console_scripts": ["trainy = trainy.cli:trainy"],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    project_urls={
        "Homepage": "https://github.com/Trainy-ai/trainy",
    },
)
