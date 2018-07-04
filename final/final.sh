#!/bin/bash
python3 preproc.py -m val -i $2
python3 test.py -m $1 -i $3 -o $4
