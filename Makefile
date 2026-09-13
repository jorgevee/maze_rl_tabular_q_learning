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
.PHONY: all render3d render3d-policy train3d ppo-demo-video policy benchmark generalization generalization-procedural generalization-random-goals generalization-random-goals-sep10 generalization-wide-conv generalization-sep-sweep ppo ppo-generalization video test clean

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

generalization-procedural: $(TARGET)
	./$(TARGET) --generalization --episodes 5000 --seeds 3 --seed 1 --procedural \
		--min-size 6 --max-size 12 --regen-every 1 --csv generalization_procedural.csv

generalization-random-goals: $(TARGET)
	./$(TARGET) --generalization --episodes 5000 --seeds 3 --seed 1 --random-goals \
		--csv generalization_random_goals.csv

generalization-random-goals-sep10: $(TARGET)
	./$(TARGET) --generalization --episodes 5000 --seeds 3 --seed 1 --random-goals \
		--min-separation 10 --csv generalization_random_goals_sep10.csv

generalization-wide-conv: $(TARGET)
	./$(TARGET) --generalization --episodes 5000 --seeds 10 --seed 1 --random-goals \
		--min-separation 10 --wide-conv --csv generalization_random_goals_sep10_wideconv_10seed.csv

generalization-sep-sweep: $(TARGET)
	./$(TARGET) --generalization --episodes 5000 --seeds 10 --seed 1 --random-goals \
		--min-separation 12 --csv generalization_random_goals_sep12_10seed.csv
	./$(TARGET) --generalization --episodes 5000 --seeds 10 --seed 1 --random-goals \
		--min-separation 14 --csv generalization_random_goals_sep14_10seed.csv

ppo: $(TARGET)
	./$(TARGET) --ppo --steps 60000 --seeds 5 --seed 1 --compare-dqn --csv ppo.csv

ppo-generalization: $(TARGET)
	./$(TARGET) --ppo --generalize --random-goals --min-separation 10 \
		--steps 575000 --seeds 10 --seed 1 --csv ppo_generalization_sep10.csv

render3d: $(TARGET)
	./$(TARGET) --render3d --maze 0

# Train a policy for the 3D view to replay, then watch it
policy: $(TARGET)
	@mkdir -p policies
	./$(TARGET) --ppo --generalize --random-goals --min-separation 10 \
		--steps 575000 --seeds 1 --seed 1 \
		--save-policy policies/ppo_sep10_seed1.bin --csv ppo_policy_run.csv

train3d: $(TARGET)
	./$(TARGET) --render3d --train --maze 16

render3d-policy: $(TARGET)
	./$(TARGET) --render3d --maze 16 --policy policies/ppo_sep10_seed1.bin

ppo-demo-video: $(TARGET)
	./scripts/render_ppo_demo.sh assets/ppo_generalization.mp4

video: $(TARGET)
	./scripts/render_generalization_video.sh assets/conv_learning.mp4

clean:
	rm -f $(TARGET) $(TEST_TARGET)
