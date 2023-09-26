#!/usr/bin/python3

import importlib

dataset_generator = importlib.import_module('dataset-generator')
letter_classifier = importlib.import_module('letter-classifier')
font_classifier = importlib.import_module('font-classifier')

print("generating dataset")
dataset_generator.main((32, 32))
print("letter classifier training")
letter_classifier.main((32, 32))
print("font classifier training")
font_classifier.main((32, 32))
