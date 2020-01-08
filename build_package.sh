#!/usr/bin/bash

rm -f -r build/*
rm -f -r dist/*

python setup.py sdist bdist_wheel
