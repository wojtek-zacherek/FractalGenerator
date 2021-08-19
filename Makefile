all: run.bin cleanO

run.o: runner.c
	gcc -Wall  -g -c runner.c -o run.o

run.bin: run.o
	gcc run.o -o run.bin -lm

cleanO:
	rm -f *.o

clean:
	rm -f *.o *.bin
	