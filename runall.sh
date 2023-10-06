#!/bin/bash

class1=$(python3 -c 'import string; print(string.ascii_lowercase)')
class2=$(python3 -c 'import string; print(string.ascii_uppercase)')
class3=$(python3 -c 'import string; print(string.digits)')

./dataset-generator.py


for class in class1 class2
do
  for (( i=0; i<${#class}; i++ ));
  do
    ./font-classifier.py "${class:$i:1}" &
  done
  wait
done

./letter-classifier.py &
for (( i=0; i<${#class3}; i++ ));
  do
    ./font-classifier.py "${class3:$i:1}" &
  done
wait
./copyresults.sh
