#!/bin/bash

# Download directories vars
root_dl="k400"
root_dl_targz="k400_targz"

# Make root directories
[ ! -d $root_dl ] && mkdir $root_dl
[ ! -d $root_dl_targz ] && mkdir $root_dl_targz

# Download train tars, will resume
curr_dl=${root_dl_targz}/train
url=https://s3.amazonaws.com/kinetics/400/train/k400_train_path.txt
[ ! -d $curr_dl ] && mkdir -p $curr_dl
wget -c -i $url -P $curr_dl

