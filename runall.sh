#!/bin/bash

class1=$(python3 -c 'import string; print(string.ascii_lowercase)')
class2=$(python3 -c 'import string; print(string.ascii_uppercase)')
class3=$(python3 -c 'import string; print(string.digits)')

./dataset-generator.py
./letter-classifier.py &
for class in class1, class2, class3
do
  for (( i=0; i<${#class}; i++ ));
  do
    ./font-classifier.py "${class:$i:1}" &
  done
  wait
done
wait
./copyresults.sh
