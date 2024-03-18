#!/bin/bash

module load python
python -W ignore plot.py $1 $2
module unload python