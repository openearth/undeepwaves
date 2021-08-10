#!/bin/sh
module load netcdf
module load mpich

echo "ld path" $LD_LIBRARY_PATH

export PATH=$PATH:~/opt/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:~/opt/lib 
source /home/deltares/venvs/main/bin/activate 

# python register_run.py


export OMP_NUM_THREADS=4

### Some general information available via SGE.
echo ----------------------------------------------------------------------
echo Run of
echo $swan_omp_exe
echo with OpenMP on linux-cluster.
echo SGE_O_WORKDIR : $SGE_O_WORKDIR
echo HOSTNAME : $HOSTNAME
echo OMP_NUM_THREADS : $OMP_NUM_THREADS
echo ----------------------------------------------------------------------

### General, start SWAN.
cp "$1".swn INPUT

mpirun -np 8 swan.exe
echo simulation ended 
rm swaninit

cp PRINT "$1".prt 
###rm PRINT 
