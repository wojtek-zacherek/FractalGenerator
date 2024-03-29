# all: run.bin runInt.bin cleanO
all: run.bin runCUDA.bin cleanO

run.o: runner.c
	gcc -Wall -g -c runner.c -o run.o


# runInt.o: runnerInt.c
# 	gcc -Wall  -g -c runnerInt.c -o runInt.o

run.bin: run.o
	gcc -pthread run.o -o run.bin -lm

runCUDA.bin: runnerCUDA.cu
	nvcc runnerCUDA.cu -o runCUDA.bin

# runInt.bin: runInt.o
# 	gcc runInt.o -o runInt.bin -lm

cleanO:
	rm -f *.o

clean:
	rm -f *.o *.bin *.png *.jpg *.pgm *.jpeg
	
