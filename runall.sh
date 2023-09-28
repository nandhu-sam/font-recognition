#!/bin/bash

./dataset-generator.py
./letter-classifier.py &
./font-classifier.py &
wait
./copyresults.sh
