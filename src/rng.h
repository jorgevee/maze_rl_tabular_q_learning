#ifndef RNG_H
#define RNG_H

#include <stdint.h>

typedef struct {
    uint64_t state;
} Rng;

void RngSeed(Rng *rng, uint64_t seed);
uint32_t RngNext(Rng *rng);
float RngFloat(Rng *rng);
int RngRange(Rng *rng, int upperExclusive);
float RngNormal(Rng *rng);

#endif
