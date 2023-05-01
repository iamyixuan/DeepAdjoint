#!/bin/bash

# Define the range of values
min=0.00001
max=0.01
n=20
# Generate a random number in the range

for (( i=1; i<=$n; i++ ))
do
    range=$(echo "$max - $min" | bc -l)
    nu=$(echo "$min + ($RANDOM / 32767 * $range)" | bc -l)
    echo "Generating data with nu equal to $nu"
    python BurgersDataGen.py -NU $nu
done
