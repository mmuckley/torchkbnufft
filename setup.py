import pathlib

from setuptools import setup, find_packages

# https://github.com/williamFalcon/pytorch-lightning/blob/master/setup.py

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

builtins.__TORCHKBNUFFT_SETUP__ = True

import torchkbnufft  # noqa: E402

install_requires = ["torch>=1.7", "numpy", "scipy"]

# https://packaging.python.org/discussions/install-requires-vs-requirements
setup(
    name="torchkbnufft",
    version=torchkbnufft.__version__,
    description=torchkbnufft.__docs__,
    author=torchkbnufft.__author__,
    author_email=torchkbnufft.__author_email__,
    url=torchkbnufft.__homepage__,
    download_url="https://github.com/mmuckley/torchkbnufft",
    license=torchkbnufft.__license__,
    packages=find_packages(exclude=["tests", "notebooks", "docs"]),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
    keywords=["MRI", "pytorch", "nufft", "deep-learning"],
    python_requires=">=3.5",
    setup_requires=[],
    install_requires=install_requires,
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
)
