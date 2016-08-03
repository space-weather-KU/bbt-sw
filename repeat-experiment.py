#!/usr/bin/env python

import os,shutil

shutil.rmtree('goes-to-goes-forecast')
for i in range(10):
    print "trial ", i
    os.remove("goes-to-goes-forecast/model.save")
    os.remove("goes-to-goes-forecast/state.save")

# i=1
# while [ $i -lt 11 ]; do
#     echo "trial $i"
#     i=`expr $i + 1`
#     rm -rf goes-to-goes-forecast/*.save
#     python goes-to-goes-forecast.py
# done
