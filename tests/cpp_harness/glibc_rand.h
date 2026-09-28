/* Force-included into the C++ build (-include glibc_rand.h) so rand()/srand() are glibc's on any platform.
 * On macOS the libc rand() is a different generator; on Linux/glibc this changes nothing. */
#ifndef GLIBC_RAND_SHIM_H
#define GLIBC_RAND_SHIM_H
#include <stdlib.h>
#ifdef __cplusplus
#include <cstdlib>
#include <vector>   /* stdafx.h uses std::vector but only got it through the CPLEX headers */
#include <climits>
extern "C" {
#endif
int glibc_rand(void);
void glibc_srand(unsigned int seed);
#ifdef __cplusplus
}
namespace std { using ::glibc_rand; using ::glibc_srand; }
#endif
#define rand glibc_rand
#define srand glibc_srand
#endif
