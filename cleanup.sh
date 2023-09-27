#!/bin/bash

if [ -d dataset ]
then rm -rf dataset
fi

if [ -d letter-clf-model ]
then rm -rf letter-clf-model
fi

if [ -d font-clf-models ]
then rm -rf font-clf-models
fi

if [ -d __pycache__ ]
then rm -rf __pycache__
fi
