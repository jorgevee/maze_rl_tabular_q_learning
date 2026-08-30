CC := gcc
CFLAGS := -std=c11 -O2 -Wall -Wextra -pedantic

ifeq ($(OS),Windows_NT)
RAYLIB_INCLUDE ?= C:/msys64/mingw64/include
RAYLIB_LIB ?= C:/msys64/mingw64/lib
LDLIBS := -L$(RAYLIB_LIB) -lraylib -lopengl32 -lgdi32 -lwinmm -lm
TARGET := maze_rl.exe
TEST_TARGET := test_rl.exe
else
RAYLIB_PREFIX ?= /opt/homebrew
RAYLIB_INCLUDE ?= $(RAYLIB_PREFIX)/include
RAYLIB_LIB ?= $(RAYLIB_PREFIX)/lib
LDLIBS := -L$(RAYLIB_LIB) -lraylib -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo -lm
TARGET := maze_rl
TEST_TARGET := test_rl
endif

CPPFLAGS := -I$(RAYLIB_INCLUDE) -Isrc

SOURCES := $(wildcard src/*.c)
LIB_SOURCES := $(filter-out src/main.c src/benchmark.c,$(SOURCES))
.PHONY: all benchmark generalization video test clean

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

video: $(TARGET)
	./scripts/render_generalization_video.sh assets/conv_learning.mp4

clean:
	rm -f $(TARGET) $(TEST_TARGET)
