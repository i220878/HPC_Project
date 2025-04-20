CC = nvcc
CFLAGS = -O2 -std=c++14 -arch=sm_70

EXE = nn
SRC = nn.cu

all: $(EXE) run

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $(EXE) $(SRC) -lm

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE)
