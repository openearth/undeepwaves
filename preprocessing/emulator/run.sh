#!/bin/bash

echo "create swan runs"
module load python37
ls /home/deltares/venvs/
source /home/deltares/venvs/main/bin/activate

cd scripts
python create_sims.py

