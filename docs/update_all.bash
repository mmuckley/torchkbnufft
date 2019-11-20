#/bin/bash

sphinx-apidoc --templatedir=source/_templates -f -o source/ ../torchkbnufft/
make html