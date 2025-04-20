# Makefile for compiling V3_CPU.c (CPU) and V5_att2.c (OpenACC)

CC = gcc
CFLAGS = -O2
LDFLAGS = -lm

# For OpenACC build (GCC)
ACCFLAGS = -fopenacc

# Targets
all: v3_cpu V5

v3_cpu: V3_CPU.c
	$(CC) $(CFLAGS) -o v3_cpu V3_CPU.c $(LDFLAGS)

V5: V5.c
	$(CC) $(CFLAGS) $(ACCFLAGS) -o V5 V5.c $(LDFLAGS)

clean:
	rm -f v3_cpu V5

.PHONY: all clean
