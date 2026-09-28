/* Independent re-implementation of glibc random_r.c TYPE_3 (pointer form, as in glibc). */
#include <stdint.h>
static int32_t randtbl[31];
static int32_t *fptr, *rptr, *end_ptr;
static int32_t next_val(void) {
    uint32_t val = *fptr += (uint32_t)*rptr;
    int32_t result = val >> 1;
    ++fptr;
    if (fptr >= end_ptr) { fptr = randtbl; ++rptr; }
    else { ++rptr; if (rptr >= end_ptr) rptr = randtbl; }
    return result;
}
void glibc_srand(unsigned int seed) {
    if (seed == 0) seed = 1;
    randtbl[0] = (int32_t)seed;
    int32_t word = (int32_t)seed;
    for (int i = 1; i < 31; ++i) {
        long hi = word / 127773, lo = word % 127773;
        word = 16807 * lo - 2836 * hi;
        if (word < 0) word += 2147483647;
        randtbl[i] = word;
    }
    fptr = &randtbl[3]; rptr = &randtbl[0]; end_ptr = &randtbl[31];
    for (int i = 0; i < 310; ++i) (void)next_val();
}
int glibc_rand(void) { return next_val(); }
