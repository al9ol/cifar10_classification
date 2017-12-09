#!/bin/bash

for file in `find tunable_params -type f -name "*.yaml"`
do
   python main.py -yaml $file
done
