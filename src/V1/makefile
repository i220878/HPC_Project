CC = gcc
CFLAGS = -Wall -pg -O2

EXE = nn.exe
SRC = nn.c

all: $(EXE) run

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $(EXE) $(SRC) -lm

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE)
