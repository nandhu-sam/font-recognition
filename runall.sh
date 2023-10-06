#!/bin/bash

classes=$(python3 -c 'import string; print(string.ascii_letters + string.digits)')
./dataset-generator.py
./letter-classifier.py &
for (( i=0; i<${#classes}; i++ ));
do
  ./font-classifier.py "${classes:$i:1}" &
done
wait
./copyresults.sh
)