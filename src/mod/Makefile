obj-m += perceptron_module.o

perceptron_module-objs := perceptron_mod.o perceptron.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
