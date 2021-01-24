#/bin/bash

sphinx-build -b html source/ build/
make clean
make html