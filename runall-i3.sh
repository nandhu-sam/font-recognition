#!/bin/bash

c=$(python3 -c 'import string; print(string.ascii_letters + string.digits)')

./dataset-generator.py
./letter-classifier.py

for (( i=0; i<${#c}; i++ ));
do
    ./font-classifier.py "${c:$i:1}"
done

./copyresults.sh

tar -czvf plots.tar.gz evaluation-plots
