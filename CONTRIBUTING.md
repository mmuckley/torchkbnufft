# Contributing Guide

`torchkbnufft` welcomes contributions!

## Submitting Code

To submit code, take the following steps:

1. Fork the repository and make your changes.
2. Add unit tests for any new functions or modules.
3. Ensure your test suite passes.
4. Make sure your code lints - we use [`black`](https://black.readthedocs.io/en/stable/),
[`flake8`](https://flake8.pycqa.org/en/latest/), and
[`mypy`](https://mypy.readthedocs.io/en/stable/).
5. Add documentation. If you've added new functions that are exposed under
`torchkbnufft`, ensure that they're properly linked in the `docs/` folder.
6. Submit a [Pull Request](https://github.com/mmuckley/torchkbnufft/pulls) to
bring your code into master!

We continuously check linting and unit tests using GitHub integrations for all
Pull Requests. If your code fails any of the tests, make the appropriate
changes and the tool will automatically retest your code.

## Issues

If you identify a bug or have a question, please fill out a GitHub issue and I
will address it as soon as I can.

## License

By submitting your code to the repository, you agree to have your submission
licensed under the terms in the LICENSE file at the root directory of this
source tree.
