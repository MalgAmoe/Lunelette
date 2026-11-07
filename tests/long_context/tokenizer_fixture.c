/* Reuse the maintained synthetic Tekken fixture, never downloaded token data. */
#define main unused_tokenizer_test_main
#include "tests/test_tokenizer.c"
#undef main

int main(void)
{
    synthetic_file(0);
    return 0;
}
