# Makefile for compiling V3_CPU.c (CPU) and V5_att2.c (OpenACC)

CC = nvc
CFLAGS = -O2
LDFLAGS = -lm

# For OpenACC build (GCC)
ACCFLAGS = -acc

# Targets
all: V5

V5: V5.c
	$(CC) $(CFLAGS) $(ACCFLAGS) -o V5 V5.c $(LDFLAGS)

clean:
	rm -f V5

.PHONY: all clean
