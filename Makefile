CC := gcc
CFLAGS := -std=c11 -O2 -Wall -Wextra -pedantic
RAYLIB_INCLUDE := C:/msys64/mingw64/include
RAYLIB_LIB := C:/msys64/mingw64/lib
LDLIBS := -L$(RAYLIB_LIB) -lraylib -lopengl32 -lgdi32 -lwinmm -lm
CPPFLAGS := -I$(RAYLIB_INCLUDE) -Isrc

SOURCES := $(wildcard src/*.c)
LIB_SOURCES := $(filter-out src/main.c src/benchmark.c,$(SOURCES))
TARGET := maze_rl.exe
TEST_TARGET := test_rl.exe

.PHONY: all benchmark generalization test clean

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS)

$(TEST_TARGET): tests/test_rl.c $(LIB_SOURCES)
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

benchmark: $(TARGET)
	./$(TARGET) --benchmark --agent both --episodes 5000 --seeds 10 --seed 1 --csv comparison.csv

generalization: $(TARGET)
	./$(TARGET) --generalization --episodes 5000 --seeds 3 --seed 1 --csv generalization.csv

clean:
	rm -f $(TARGET) $(TEST_TARGET)
