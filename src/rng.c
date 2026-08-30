#include "rng.h"
#include <math.h>

#define RL_PI 3.14159265358979323846f

void RngSeed(Rng *rng, uint64_t seed)
{
    rng->state = seed != 0 ? seed : UINT64_C(0x9e3779b97f4a7c15);
}

uint32_t RngNext(Rng *rng)
{
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return (uint32_t)((x * UINT64_C(2685821657736338717)) >> 32);
}

float RngFloat(Rng *rng)
{
    return (float)(RngNext(rng) >> 8) * (1.0f / 16777216.0f);
}

int RngRange(Rng *rng, int upperExclusive)
{
    if (upperExclusive <= 1) return 0;
    return (int)(RngNext(rng) % (uint32_t)upperExclusive);
}

float RngNormal(Rng *rng)
{
    float u1 = RngFloat(rng);
    float u2 = RngFloat(rng);
    if (u1 < 1.0e-7f) u1 = 1.0e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * RL_PI * u2);
}
