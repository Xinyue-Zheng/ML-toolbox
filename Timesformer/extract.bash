#!/bin/bash

# Download directories vars
root_dl="/home/xinyue/kinetics-dataset/k400"
root_dl_targz="/home/xinyue/kinetics-dataset/k400_targz"

# Make root directories
[ ! -d $root_dl ] && mkdir $root_dl

# Extract train
curr_dl=$root_dl_targz/train
curr_extract=$root_dl/train
[ ! -d $curr_extract ] && mkdir -p $curr_extract
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract
done