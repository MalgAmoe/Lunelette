/* Build-time text embedding: ISO C11, no shader compiler or Python required. */
#include <stdio.h>

int main(int argc, char **argv)
{
    if (argc != 2) {
        return 1;
    }
    FILE *input = fopen(argv[1], "rb");
    if (!input) {
        perror(argv[1]);
        return 1;
    }
    puts("/* Generated from src/metal_kernels.metal. */");
    puts("static const unsigned char lunelette_metal_shader[] = {");
    size_t column = 0;
    int byte;
    while ((byte = fgetc(input)) != EOF) {
        printf("%u,", (unsigned)byte);
        if (++column == 24) {
            putchar('\n');
            column = 0;
        }
    }
    puts("0\n};");
    int failed = ferror(input) || ferror(stdout);
    if (fflush(stdout) != 0) {
        failed = 1;
    }
    if (fclose(input) != 0) {
        failed = 1;
    }
    return failed ? 1 : 0;
}
