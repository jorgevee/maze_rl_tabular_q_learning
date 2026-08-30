#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <stdbool.h>
#include <stdint.h>
#include "rl.h"

typedef struct {
    bool runTabular;
    bool runDqn;
    int episodes;
    int seedCount;
    uint64_t firstSeed;
    const char *csvPath;
} BenchmarkOptions;

BenchmarkOptions DefaultBenchmarkOptions(void);
bool ParseBenchmarkOptions(int argc, char **argv, BenchmarkOptions *options);
int RunBenchmark(const BenchmarkOptions *options);

#endif
