#include "benchmark.h"
#include "dqn.h"
#include "tabular.h"
#include "trainer.h"
#include "agent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

BenchmarkOptions DefaultBenchmarkOptions(void)
{
    return (BenchmarkOptions){true, true, 5000, 10, 1, "comparison.csv"};
}

static bool ParsePositive(const char *text, int *value)
{
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 1 || parsed > 1000000) return false;
    *value = (int)parsed;
    return true;
}

bool ParseBenchmarkOptions(int argc, char **argv, BenchmarkOptions *options)
{
    *options = DefaultBenchmarkOptions();
    bool foundBenchmark = false;
    for (int index = 1; index < argc; index++) {
        if (strcmp(argv[index], "--benchmark") == 0) foundBenchmark = true;
        else if (strcmp(argv[index], "--agent") == 0 && index + 1 < argc) {
            const char *agent = argv[++index];
            options->runTabular = strcmp(agent, "tabular") == 0 || strcmp(agent, "both") == 0;
            options->runDqn = strcmp(agent, "dqn") == 0 || strcmp(agent, "both") == 0;
            if (!options->runTabular && !options->runDqn) return false;
        } else if (strcmp(argv[index], "--episodes") == 0 && index + 1 < argc) {
            if (!ParsePositive(argv[++index], &options->episodes)) return false;
        } else if (strcmp(argv[index], "--seeds") == 0 && index + 1 < argc) {
            if (!ParsePositive(argv[++index], &options->seedCount)) return false;
        } else if (strcmp(argv[index], "--seed") == 0 && index + 1 < argc) {
            int seed;
            if (!ParsePositive(argv[++index], &seed)) return false;
            options->firstSeed = (uint64_t)seed;
        } else if (strcmp(argv[index], "--csv") == 0 && index + 1 < argc) {
            options->csvPath = argv[++index];
        } else if (strcmp(argv[index], "--benchmark") != 0) {
            return false;
        }
    }
    return foundBenchmark;
}

static bool CreateForKind(AgentKind kind, Learner *learner, Rng *rng)
{
    if (kind == AGENT_DQN) return CreateDqnLearner(learner, DefaultDqnConfig(), rng);
    return CreateTabularLearner(learner, 0.1f, 0.95f);
}

static void WriteRow(
    FILE *csv,
    uint64_t seed,
    const Learner *learner,
    const TrainingStats *stats,
    EpisodeResult evaluation,
    double elapsedMilliseconds)
{
    fprintf(csv, "%llu,%s,%d,%.3f,%.2f,%.2f,%d,%d,%.1f,%.3f,%zu,%zu\n",
        (unsigned long long)seed,
        AgentKindName(learner->kind),
        stats->episodes,
        stats->lastEpisode.totalReward,
        stats->successRate,
        stats->averageSuccessfulSteps,
        evaluation.reachedGoal ? 1 : 0,
        evaluation.steps,
        evaluation.totalReward,
        elapsedMilliseconds,
        LearnerParameterCount(learner),
        LearnerMemoryBytes(learner));
}

static int RunOne(FILE *csv, AgentKind kind, int episodes, uint64_t seed)
{
    Rng trainingRng;
    RngSeed(&trainingRng, seed ^ ((uint64_t)kind << 48));
    Learner learner = {0};
    if (!CreateForKind(kind, &learner, &trainingRng)) return 1;
    TrainingStats stats;
    InitializeTrainingStats(&stats);
    clock_t start = clock();

    Rng evaluationRng;
    RngSeed(&evaluationRng, seed ^ UINT64_C(0xa5a5a5a5));
    EpisodeResult evaluation = EvaluateGreedy(&learner, &evaluationRng);
    WriteRow(csv, seed, &learner, &stats, evaluation, 0.0);

    while (stats.episodes < episodes) {
        TrainAndRecordEpisode(&learner, &stats, &trainingRng);
        if (stats.episodes % 100 == 0 || stats.episodes == episodes) {
            RngSeed(&evaluationRng, seed ^ (uint64_t)stats.episodes ^ UINT64_C(0xa5a5a5a5));
            evaluation = EvaluateGreedy(&learner, &evaluationRng);
            double elapsed = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
            WriteRow(csv, seed, &learner, &stats, evaluation, elapsed);
        }
    }
    printf("seed=%llu agent=%s episodes=%d greedy_success=%s greedy_steps=%d\n",
        (unsigned long long)seed, AgentKindName(kind), episodes,
        evaluation.reachedGoal ? "yes" : "no", evaluation.steps);
    DestroyLearner(&learner);
    return 0;
}

int RunBenchmark(const BenchmarkOptions *options)
{
    FILE *csv = fopen(options->csvPath, "w");
    if (csv == NULL) {
        fprintf(stderr, "Could not open CSV output: %s\n", options->csvPath);
        return 1;
    }
    fprintf(csv, "seed,agent,episode,training_return,rolling_success_pct,avg_success_steps,greedy_success,greedy_steps,greedy_return,elapsed_ms,parameters,memory_bytes\n");
    int status = 0;
    for (int offset = 0; offset < options->seedCount && status == 0; offset++) {
        uint64_t seed = options->firstSeed + (uint64_t)offset;
        if (options->runTabular) status = RunOne(csv, AGENT_TABULAR, options->episodes, seed);
        if (status == 0 && options->runDqn) status = RunOne(csv, AGENT_DQN, options->episodes, seed);
        fflush(csv);
    }
    fclose(csv);
    if (status == 0) printf("Wrote benchmark metrics to %s\n", options->csvPath);
    return status;
}
