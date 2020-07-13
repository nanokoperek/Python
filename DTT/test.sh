#!/bin/bash
for i in {0..150}
do
	echo $i
	python DTT_generator.py
done
