#!/bin/sh

# $1 scratch folder
# $2 train list
# $3 test list
# $4 output list

# encoding feature
python run_svm.py "$1" "$2" "$3" "$4" 


