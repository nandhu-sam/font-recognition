#!/bin/bash

if [ -d evaluation-plots ]
then rm -rf evaluation-plots
fi

mkdir -p evaluation-plots/accuracy
mkdir -p evaluation-plots/loss
mkdir -p evaluation-plots/confusionmatrix

find . | grep 'accuracy.*\.svg$' | xargs cp -t evaluation-plots/accuracy
find . | grep 'loss.*\.svg$' | xargs cp -t evaluation-plots/loss
find . | grep 'confusionmatrix.*\.svg$' | xargs cp -t evaluation-plots/confusionmatrix
