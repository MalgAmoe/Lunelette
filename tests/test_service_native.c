#include "native.h"
#include "metal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

/* CPU-only adapter contract test. Link-time runtime stubs, never production.
 * argv[1] is a tokenizer, NOT weights; create does not open a checkpoint. */
struct lunelette_metal {
    int64_t ids[256];
    unsigned count;
};
static unsigned creates, destroys, prefills, decodes, resets;
static int fail;
static float logits[131072];
lunelette_metal *lunelette_metal_create_with_options(const char *path,
    const lunelette_metal_options *options, char *error, size_t capacity)
{
    (void)path;
    (void)error;
    (void)capacity;
    CHECK(options->prefill == LUNELETTE_METAL_PREFILL_CUSTOM);
    creates++;
    return calloc(1, sizeof(lunelette_metal));
}
void lunelette_metal_destroy(lunelette_metal *runtime)
{
    destroys++;
    free(runtime);
}
int lunelette_metal_reset(lunelette_metal *runtime, char *error, size_t capacity)
{
    (void)error;
    (void)capacity;
    resets++;
    runtime->count = 0;
    return 0;
}
static int result(const float **out, char *error, size_t capacity)
{
    if (fail) {
        *out = NULL;
        snprintf(error, capacity, "deliberate test failure");
        return -1;
    }
    memset(logits, 0, sizeof(logits));
    /* Equal maxima must choose the lower ID using maintained policy. */
    logits[22177] = logits[22178] = 1;
    *out = logits;
    return 0;
}
int lunelette_metal_prefill(lunelette_metal *runtime, const int64_t *ids, size_t count,
                           const float **out, char *error, size_t capacity)
{
    CHECK(!runtime->count && count && count <= 256);
    memcpy(runtime->ids, ids, count * sizeof(*ids));
    runtime->count = (unsigned)count;
    prefills++;
    return result(out, error, capacity);
}
int lunelette_metal_decode(lunelette_metal *runtime, int64_t token,
                          const float **out, char *error, size_t capacity)
{
    CHECK(runtime->count && runtime->count < 256);
    runtime->ids[runtime->count++] = token;
    decodes++;
    return result(out, error, capacity);
}
int main(int argc, char **argv)
{
    CHECK(argc == 2);
    char *arguments[] = {"native-test", "--model", argv[1], "--tokenizer", argv[1],
        "--strategy", "resident-fp32", "--context", "256", "--prefill-chunk", "256",
        "--prefill", "custom", "--port", "0"};
    service_generator generator;
    unsigned port;
    CHECK(!service_native_open(1, arguments, &generator, &port));
    CHECK(creates == 0);
    const char *invalid_ports[] = {"-0", "+0", " 0", "65536", "18446744073709551616"};
    for (unsigned i = 0; i < sizeof(invalid_ports) / sizeof(invalid_ports[0]); i++) {
        arguments[14] = (char *)invalid_ports[i];
        CHECK(!service_native_open(15, arguments, &generator, &port));
        CHECK(creates == 0);
    }
    arguments[14] = "0";
    CHECK(service_native_open(15, arguments, &generator, &port));
    CHECK(creates == 1 && port == 0);
    CHECK(strstr(generator.info, "\"model_load_count\":1"));
    unsigned ids[256], count = 256;
    CHECK(generator.frame(generator.context, (const unsigned char *)"Hi", 2, 1, ids, &count));
    CHECK(count == 4 && ids[0] == 1 && ids[2] == 37133);
    CHECK(generator.materialize(generator.context, ids, count));
    CHECK(prefills == 1 && decodes == 0);
    CHECK(generator.next(generator.context) == 22177);
    CHECK(generator.next(generator.context) == -1);
    CHECK(!generator.materialize(generator.context, ids, count));
    ids[count++] = 22177;
    CHECK(generator.materialize(generator.context, ids, count));
    CHECK(prefills == 1 && decodes == 1);
    CHECK(generator.materialize(generator.context, ids, count));
    CHECK(decodes == 1);
    ids[2]++;
    CHECK(!generator.materialize(generator.context, ids, count));
    ids[2]--;
    /* A generated EOS remains pending until new framed user input arrives. */
    logits[2] = 2;
    CHECK(generator.next(generator.context) == 2);
    ids[count++] = 2;
    unsigned capacity = 256 - count;
    CHECK(generator.frame(generator.context, (const unsigned char *)"Again", 5, 0,
                          ids + count, &capacity));
    count += capacity;
    CHECK(generator.materialize(generator.context, ids, count));
    CHECK(prefills == 2 && decodes == 1);
    CHECK(generator.next(generator.context) == 22177);
    unsigned char bytes[1024];
    size_t length = sizeof(bytes);
    CHECK(generator.piece(generator.context, 22177, bytes, &length));
    CHECK(length == 5 && !memcmp(bytes, "Hello", 5));
    length = sizeof(bytes);
    CHECK(!generator.piece(generator.context, 17, bytes, &length));
    ids[count++] = 22177;
    fail = 1;
    CHECK(!generator.materialize(generator.context, ids, count));
    CHECK(generator.next(generator.context) == -1);
    fail = 0;
    generator.reset(generator.context);
    count = 256;
    CHECK(generator.frame(generator.context, (const unsigned char *)"Hi", 2, 1, ids, &count));
    CHECK(generator.materialize(generator.context, ids, count));
    CHECK(generator.next(generator.context) == 22177);
    /* Full-vocabulary nonfinite check is inherited, not a sampled check. */
    ids[count++] = 22177;
    CHECK(generator.materialize(generator.context, ids, count));
    logits[131071] = NAN;
    CHECK(generator.next(generator.context) == -1);
    generator.reset(generator.context);
    generator.destroy(generator.context);
    CHECK(creates == 1 && destroys == 1 && prefills == 3 && resets >= 4);
    puts("native adapter CPU stub contract passed (not model validation)");
    return 0;
}
