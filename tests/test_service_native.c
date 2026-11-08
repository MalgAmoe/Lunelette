#include "support/native_runtime.h"

static void handle_test(void)
{
    lunelette_metal runtime = {.capacity = 256, .block = 256};
    const int64_t ids[] = {42, 43};
    const float *out;
    char error[128];
    lunelette_metal_checkpoint *old, *replacement;
    CHECK(!lunelette_metal_prefill(&runtime, ids, 2, &out, error, sizeof(error)));
    CHECK(!lunelette_metal_checkpoint_create(&runtime, &old, error, sizeof(error)));
    fail_snapshot = 1;
    CHECK(lunelette_metal_checkpoint_create(&runtime, &replacement, error, sizeof(error)));
    fail_snapshot = 0;
    CHECK(!lunelette_metal_checkpoint_restore(&runtime, old, &out, error, sizeof(error)));
    CHECK(!lunelette_metal_checkpoint_create(&runtime, &replacement, error, sizeof(error)));
    CHECK(lunelette_metal_checkpoint_restore(&runtime, old, &out, error, sizeof(error)));
    CHECK(!out);
    lunelette_metal_checkpoint_destroy(old);
    CHECK(!lunelette_metal_checkpoint_restore(&runtime, replacement, &out, error, sizeof(error)));
    CHECK(!lunelette_metal_reset(&runtime, error, sizeof(error)));
    CHECK(!lunelette_metal_prefill(&runtime, ids, 2, &out, error, sizeof(error)));
    CHECK(lunelette_metal_checkpoint_restore(&runtime, replacement, &out, error, sizeof(error)));
    lunelette_metal_checkpoint_destroy(replacement);
    CHECK(!lunelette_metal_checkpoint_create(&runtime, &replacement, error, sizeof(error)));
    fail = 1;
    CHECK(lunelette_metal_decode(&runtime, 44, &out, error, sizeof(error)));
    fail = 0;
    CHECK(lunelette_metal_checkpoint_restore(&runtime, replacement, &out, error, sizeof(error)));
    CHECK(!out);
    lunelette_metal_checkpoint_destroy(replacement);
}

static void policy_test(char **arguments, unsigned policy)
{
    char *args[17];
    memcpy(args, arguments, 15 * sizeof(*args));
    args[15] = "--prefix-reuse";
    args[16] = policy == 1 ? "on" : "off";
    service_generator g;
    unsigned port;
    CHECK(service_native_open(17, args, &g, &port));
    CHECK(g.prefix_reuse == policy);
    CHECK(strstr(g.info, "\"prefix_reuse_validation\":\"external_evidence_only\""));
    CHECK(strstr(g.info, "published source-bound evidence; no runtime attestation"));
    CHECK(strstr(g.info, "\"numerical_comparison_performed_at_startup\":false"));
    CHECK(!strstr(g.info, "NOT_RUN"));
    unsigned ids[256], length = 100;
    for (unsigned i = 0; i < length; i++) {
        ids[i] = 42;
    }
    service_work w;
    g.begin(g.context);
    CHECK(g.materialize(g.context, ids, length));
    g.work(g.context, &w);
    CHECK(w.full_committed == 100 && w.prefill_calls == 1);
    CHECK(g.save(g.context) == 1);
    unsigned before = appends, full = prefills, scalar = decodes;
    for (unsigned i = 0; i < 12; i++) {
        ids[length++] = 43;
    }
    g.begin(g.context);
    CHECK(g.materialize(g.context, ids, length));
    g.work(g.context, &w);
    CHECK(w.full_committed == (policy == 1 ? 0 : 112));
    CHECK(w.suffix_committed == (policy == 1 ? 12 : 0));
    CHECK(appends == before + (policy == 1));
    CHECK(prefills == full + (policy == 2));
    CHECK(g.next(g.context) == 22177);
    ids[length++] = 22177;
    CHECK(g.materialize(g.context, ids, length));
    g.work(g.context, &w);
    CHECK(w.decode_calls == 1 && w.pending_committed == 1 && w.decode_committed == 1);
    CHECK(decodes == scalar + 1); /* off does not recompute every decode */
    for (unsigned repeat = 0; repeat < 3; repeat++) {
        full = prefills;
        scalar = decodes;
        CHECK(g.restore(g.context) == 1);
        g.work(g.context, &w);
        CHECK(w.logit_restores == (policy == 1));
        length = 100;
        CHECK(prefills == full && decodes == scalar);
        g.begin(g.context);
        CHECK(g.materialize(g.context, ids, length));
        g.work(g.context, &w);
        CHECK(w.full_committed == (policy == 2 ? 100 : 0));
        CHECK(w.decode_calls == 0);
        CHECK(g.next(g.context) == 22177);
        ids[length++] = 22177;
        CHECK(g.materialize(g.context, ids, length));
    }
    full = prefills;
    scalar = decodes;
    g.begin(g.context);
    CHECK(g.materialize(g.context, ids, length));
    g.work(g.context, &w);
    CHECK(w.full_committed == (policy == 2 ? length : 0));
    CHECK(prefills == full + (policy == 2) && decodes == scalar);
    ids[0]++;
    CHECK(!g.materialize(g.context, ids, length));
    ids[0]--;
    /* Ordinary new turn reconciles pending EOS and user suffix together. */
    logits[2] = 2;
    CHECK(g.next(g.context) == 2);
    ids[length++] = 2;
    ids[length++] = 42;
    ids[length++] = 43;
    g.begin(g.context);
    CHECK(g.materialize(g.context, ids, length));
    g.work(g.context, &w);
    CHECK(w.pending_committed == 1);
    CHECK(w.append_calls == (policy == 1));
    CHECK(w.suffix_committed == (policy == 1 ? 3 : 0));
    /* Pending EOS materialized once, including no new user suffix. */
    logits[2] = 2;
    CHECK(g.next(g.context) == 2);
    ids[length++] = 2;
    CHECK(g.materialize(g.context, ids, length));
    g.work(g.context, &w);
    CHECK(w.pending_committed == 1 && w.decode_calls == 1);
    CHECK(g.save(g.context) == 1);
    CHECK(g.restore(g.context) == 1);
    CHECK(g.materialize(g.context, ids, length));
    /* Failed replacement retains previous snapshot. */
    fail_snapshot = 1;
    CHECK(g.save(g.context) == 0);
    fail_snapshot = 0;
    CHECK(g.restore(g.context) == 1);
    CHECK(g.materialize(g.context, ids, length));
    CHECK(g.next(g.context) == 22177);
    ids[length++] = 22177;
    fail = 1;
    CHECK(!g.materialize(g.context, ids, length));
    g.work(g.context, &w);
    CHECK(w.decode_calls == 1 && w.decode_attempted == 1 && !w.decode_committed);
    CHECK(w.pending_attempted == 1 && !w.pending_committed);
    fail = 0;
    if (policy == 1) {
        CHECK(g.restore(g.context) == 0); /* Execution failure invalidates the handle. */
    }
    g.reset(g.context);
    CHECK(g.restore(g.context) == 0);
    g.destroy(g.context);
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
    const char *strategies[] = {"bad", "", "COMPACT"};
    for (size_t i = 0; i < sizeof strategies / sizeof strategies[0]; i++) {
        arguments[6] = (char *)strategies[i];
        CHECK(!service_native_open(15, arguments, &generator, &port));
        CHECK(creates == 1);
    }
    arguments[6] = "compact-fp8-arithmetic";
    char *duplicate[17];
    memcpy(duplicate, arguments, sizeof arguments);
    duplicate[15] = "--strategy";
    duplicate[16] = "resident-fp32";
    CHECK(!service_native_open(17, duplicate, &generator, &port));
    CHECK(creates == 1);
    CHECK(service_native_open(15, arguments, &generator, &port));
    CHECK(strstr(generator.info, "\"execution_strategy\":\"compact-fp8-arithmetic\""));
    CHECK(strstr(generator.info, "\"named_conversion_stage_bytes\":113246208"));
    CHECK(strstr(generator.info, "\"named_buffer_total_bytes\":4893045760"));
    CHECK(strstr(generator.info, "\"compact_model_validation\":\"external_evidence_only\""));
    CHECK(strstr(generator.info, "\"numerical_comparison_performed_at_startup\":false"));
    CHECK(strstr(generator.info, "\"published_validation_document\":\"docs/real-workload-m4.md\""));
    CHECK(strstr(generator.info, "no runtime attestation"));
    CHECK(!strstr(generator.info, "NOT_RUN"));
    CHECK(strstr(generator.info, "FP8-convert-stage-then-precise-FP32"));
    CHECK(!strstr(generator.info, "MPS accumulation unspecified"));
    CHECK(strlen(generator.info) + 256 < 4096); /* /info process sample also fits. */
    tj_document info;
    CHECK(tj_read_memory_limits(generator.info, strlen(generator.info), &info,
        8192, 4096, TJ_MAX_NODES, 32));
    CHECK(tj_field(&info, 1, "execution_strategy") != 0);
    tj_free(&info);
    generator.destroy(generator.context);
    CHECK(creates == 2 && destroys == 2);
    expected_prefill = LUNELETTE_METAL_PREFILL_MPS;
    arguments[12] = "mps";
    CHECK(service_native_open(15, arguments, &generator, &port));
    CHECK(strstr(generator.info, "FP8-convert-stage-then-MPS"));
    CHECK(strstr(generator.info, "MPS accumulation unspecified"));
    generator.destroy(generator.context);
    fail_memory = 1;
    CHECK(!service_native_open(15, arguments, &generator, &port));
    CHECK(creates == 4 && destroys == 4); /* Getter failure closes runtime/tokenizer. */
    fail_memory = 0;
    wrong_strategy = 1;
    CHECK(!service_native_open(15, arguments, &generator, &port));
    CHECK(creates == 5 && destroys == 5); /* A valid but unrequested strategy is not fallback. */
    wrong_strategy = 0;
    arguments[6] = "compact-fp8-lookup";
    CHECK(service_native_open(15, arguments, &generator, &port));
    CHECK(strstr(generator.info, "\"execution_strategy\":\"compact-fp8-lookup\""));
    CHECK(strstr(generator.info, "compact-FP8-lookup"));
    CHECK(strstr(generator.info, "\"published_validation_document\":\"docs/fp8-lookup.md\""));
    CHECK(strstr(generator.info, "\"named_conversion_stage_bytes\":113246208"));
    generator.destroy(generator.context);
    policy_test(arguments, 1);
    policy_test(arguments, 2);
    CHECK(snapshots >= 4 && restores >= 4);
    handle_test();
    puts("native adapter CPU stub contract passed (not model validation)");
    return 0;
}
