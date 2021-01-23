#/bin/bash

export PYTORCH_JIT=0
sphinx-build -b html source/ build/
make clean
make html