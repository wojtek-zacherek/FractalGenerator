# Fractal Generator
## Overview TL;DR
This repository store my attempts at making a powerful fractal generator. As of September 10, 2021, I have a prototype working in C, using multi CPU threads to perform the calculation.

## Setup
Setup is very straight forward, especially since only the CPU threaded version is expected to run. Run the following to download
```
git clone https://github.com/wojtek-zacherek/FractalGenerator
```
Then simply `make` the c file and it should output a runnable program with a `*.bin` extension

## Running
The command takes the following parameters:
1. x & y resolution = integer values for the final pixel resolution of the image
2. threshold = how quickly a candidate point should be dropped off during calculation
3. iterations = how many times a point is tested recursively
4. ratio = readjusts the input resolution to be at a 1.7778 x/y retio.
   - 0 adjusts y
   - 1 adjusts x
Currently, to specify a parameter from the list, all the previous parameters must be defined. For example, to specify the `iterations` parameter, the user must specify the `threshold` and `resolution` parameters, but not the `ratio`.

Examples
```
./run.bin 1920 1080
./run.bin 1920 1080 8 255 1
./run.bin 1920 1080 8 255 1
```

## TODO
- Modify parameter inputs to not be sequentially dependent
- Convert to use purely integer operation
- Rewrite for CUDA