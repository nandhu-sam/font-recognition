#!/usr/bin/python3

import importlib

dataset_generator = importlib.import_module('dataset-generator')
letter_classifier = importlib.import_module('letter-classifier')
font_classifier = importlib.import_module('font-classifier')

print("generating dataset")
dataset_generator.main((64, 64))
print("letter classifier training")
letter_classifier.main((64, 64))
print("font classifier training")
font_classifier.main((64, 64), parallel=True)
