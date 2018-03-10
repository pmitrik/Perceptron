CC=gcc
CFLAGS=-I.
DEPS = perceptron.h
OBJS = example.o perceptron.o 
MAIN = perceptron

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

all:	$(MAIN)

$(MAIN): $(OBJS)
	gcc -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -f *.o *~ core $(MAIN)

