/* Build-time text embedding: ISO C11, no shader compiler or Python required. */
#include <stdio.h>
#include "../include/model_facts.h"

#define STRINGIFY_INNER(value) #value
#define STRINGIFY(value) STRINGIFY_INNER(value)

/* Emit the facts into the shader itself, not the generated host C namespace. */
static const char shader_facts[] =
    "#define LUNELETTE_MODEL_NORM_EPS " STRINGIFY(LUNELETTE_MODEL_NORM_EPS) "\n"
    "#define LUNELETTE_MODEL_QUERY_SCALE " STRINGIFY(LUNELETTE_MODEL_QUERY_SCALE) "\n"
    "#define LUNELETTE_MODEL_ORIGINAL_CONTEXT " STRINGIFY(LUNELETTE_MODEL_ORIGINAL_CONTEXT) "\n";

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
    for (size_t i = 0; i < sizeof shader_facts - 1; i++) {
        printf("%u,", (unsigned)(unsigned char)shader_facts[i]);
        if (++column == 24) {
            putchar('\n');
            column = 0;
        }
    }
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
