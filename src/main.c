#include "lunelette.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--version") == 0) {
        puts("Lunelette " LUNELETTE_VERSION);
        return 0;
    }
    if (argc == 1 || (argc == 2 && strcmp(argv[1], "--help") == 0)) {
        puts("Usage: lunelette [--help | --version]\n"
             "Ministral 3 inference in C.\n"
             "Embedding reader milestone; full inference and generation are not implemented.");
        return 0;
    }
    fputs("Unknown argument. Use --help.\n", stderr);
    return 1;
}
