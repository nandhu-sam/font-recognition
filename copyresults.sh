#!/bin/bash

if [ -d evaluation-plots ]
then rm -rf evaluation-plots
fi

mkdir -p evaluation-plots/accuracy
mkdir -p evaluation-plots/loss

find . | grep 'accuracy.*\.svg$' | xargs cp -t evaluation-plots/accuracy
find . | grep 'loss.*\.svg$' | xargs cp -it evaluation-plots/loss
