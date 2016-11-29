#!/usr/bin/env python

import os,shutil,subprocess

experiment_dir = 'goes-to-goes-forecast'

def when_path_exists(path, func):
    if os.path.exists(path):
        func(path)

when_path_exists(experiment_dir, shutil.rmtree)

for i in range(10):
    print "trial ", i
    when_path_exists(os.path.join(experiment_dir,"model.save"), os.remove)
    when_path_exists(os.path.join(experiment_dir,"state.save") ,os.remove)
    subprocess.call(["python","goes-to-goes-forecast.py"])
# i=1
# while [ $i -lt 11 ]; do
#     echo "trial $i"
#     i=`expr $i + 1`
#     rm -rf goes-to-goes-forecast/*.save
#     python goes-to-goes-forecast.py
# done
