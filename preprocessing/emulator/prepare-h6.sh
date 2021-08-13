#!/bin/bash  

# load relevant  modules
module load netcdf/v4.7.4_v4.5.3_intel19.1.1 
module load anaconda3/2021.05
module load swan

# create new anaconda environment
conda create -n undeepwave 

conda activate undeepwave

mkdir ~/download
pushd download
# download gsutil and other google sdk
wget 'https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-352.0.0-linux-x86_64.tar.gz'
tar -xzf google-cloud-sdk-352.0.0-linux-x86_64.tar.gz 
cd google-cloud-sdk/
./install.sh 
# fix gsutil issue
cp google-cloud-sdk/platform/gsutil/gslib/../CHECKSUM google-cloud-sdk/platform/gsutil/gslib
# load changed bashrc
. ~/.bashrc
popd


gsutil


mkdir -p ~/data/undeepwaves
cd ~/data/undeepwaves
# gsutil -m cp -r gs://bathy_sample/01_sims/U01D000Lp100Tz0e+00a .
# cd U01D000Lp100Tz0e+00a/
# swan_4131A_1_del_l64_i18_omp.exe
# ncdump -h  results/U01D000Lp100Tz0e+00a.nc 
