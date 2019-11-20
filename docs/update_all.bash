#/bin/bash

rm -f -r build/*
sphinx-apidoc --template_dir=source/_templates -f -o source/ ../torchkbnufft/
make html