/* Reference glibc rand() stream, used by test_cxx_compat.py to check the Python port. */
#include <stdio.h>
#include <stdlib.h>
int main(void) {
    int seeds[] = {495, 595, 1394, 0};
    for (int q = 0; q < 4; q++) {
        srand(seeds[q]);
        printf("%d:", seeds[q]);
        for (int i = 0; i < 6; i++) printf(" %d", rand());
        printf("\n");
    }
    srand(495);
    for (int i = 0; i < 999; i++) rand();
    printf("495@1000: %d\n", rand());
    for (int i = 0; i < 99000; i++) rand();
    printf("495@100001: %d\n", rand());
    return 0;
}
