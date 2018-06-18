CC = gcc
INCLUDES = -I.
CFLAGS = -Wall -g
DEPS = perceptron.h
OBJS = example.o perceptron.o
MAIN = perceptron
LFLAGS =
LIBS =

%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

all:	$(MAIN)

$(MAIN): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f *.o *~ core $(MAIN)
