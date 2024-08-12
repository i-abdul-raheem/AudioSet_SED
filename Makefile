# Makefile

train:
	python train.py

train-cached:
	python train.py --cached

predict:
	python predict.py
