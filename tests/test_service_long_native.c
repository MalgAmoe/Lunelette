/* Actual native adapter, link-time CPU runtime stub. Optional official tokenizer,
 * never weights. Include native implementation to report owned host layout. */
#define main short_native_main
#include "test_service_native.c"
#undef main
#include "../service/native.c"

static void ingest(service_generator *g, unsigned *ids, unsigned length,
                   service_work *total)
{
    unsigned cursor = ((native_service *)g->context)->length;
    int state;
    memset(total, 0, sizeof(*total));
    do {
        state = g->materialize_step(g->context, ids, length, &cursor);
        CHECK(state == 1 || state == 2);
        native_service *native = g->context;
        CHECK(native->runtime->count == cursor);
        for (unsigned i = 0; i < cursor; i++) {
            CHECK(native->runtime->ids[i] == (int64_t)ids[i]);
        }
        service_work w;
        g->work(g->context, &w);
        CHECK(w.full_committed + w.suffix_committed + w.decode_committed <= g->block_capacity);
        total->full_attempted += w.full_attempted;
        total->full_committed += w.full_committed;
        total->suffix_attempted += w.suffix_attempted;
        total->suffix_committed += w.suffix_committed;
        total->prefill_calls += w.prefill_calls;
        total->append_calls += w.append_calls;
        total->decode_calls += w.decode_calls;
        total->decode_committed += w.decode_committed;
        total->pending_committed += w.pending_committed;
        total->recompute_committed += w.recompute_committed;
    } while (state == 2);
    CHECK(cursor == length);
}

int main(int argc, char **argv)
{
    CHECK(argc == 2);
    printf("native_owner=%zu request=%zu session=%zu history=%zu\n",
           sizeof(native_service), sizeof(service_request), sizeof(service_session),
           sizeof(((service_session *)0)->history));
    for (unsigned block = 128; block <= 256; block *= 2) {
        for (unsigned policy = 0; policy < 3; policy++) {
            char *args[] = {"test", "--model", argv[1], "--tokenizer", argv[1],
                "--strategy", "resident-fp32", "--context", "4352", "--prefill-chunk",
                block == 128 ? "128" : "256", "--prefill", "custom", "--port", "0",
                "--prefix-reuse", policy == 1 ? "on" : "off"};
            service_generator g;
            unsigned port;
            CHECK(service_native_open(policy ? 17 : 15, args, &g, &port));
            CHECK(g.context_capacity == 4352 && g.block_capacity == block);
            unsigned ids[4353];
            for (unsigned i = 0; i < 4353; i++) {
                ids[i] = 1000 + (37 * i) % 8000;
            }
            service_work w;
            unsigned lengths[] = {block, block + 1, 4096, 4352};
            for (unsigned k = 0; k < 4; k++) {
                g.reset(g.context);
                unsigned reset_count = resets;
                g.begin(g.context);
                ingest(&g, ids, lengths[k], &w);
                CHECK(resets == reset_count + 1);
                CHECK(w.prefill_calls == 1 && w.full_committed == block);
                CHECK(w.append_calls == (lengths[k] - 1) / block);
                CHECK(w.full_committed + w.suffix_committed == lengths[k]);
                CHECK(w.recompute_committed == 0);
                CHECK(g.save(g.context) == 1);
            }
            unsigned cursor;
            CHECK(g.materialize_step(g.context, ids, 4353, &cursor) == 0);
            for (unsigned eos = 0; eos < 2; eos++) {
                g.reset(g.context);
                g.begin(g.context);
                ingest(&g, ids, 4096, &w);
                if (eos) {
                    logits[2] = 2;
                }
                int token = g.next(g.context);
                CHECK(token == (eos ? 2 : 22177));
                ids[4096] = (unsigned)token;
                ingest(&g, ids, 4097, &w);
                CHECK(w.pending_committed == 1);
                CHECK(w.full_committed + w.suffix_committed + w.decode_committed ==
                      (!policy && eos ? 4097 : 1));
                CHECK(g.save(g.context) == 1);
                CHECK(g.restore(g.context) == 1);
                if (policy == 2) {
                    ingest(&g, ids, 4097, &w);
                }
                CHECK(g.next(g.context) == 22177);
                ids[4097] = 22177;
                ingest(&g, ids, 4098, &w);
                CHECK(g.restore(g.context) == 1);
                ids[4097] = 43;
                g.begin(g.context);
                ingest(&g, ids, 4098, &w);
                CHECK(w.full_committed + w.suffix_committed == (policy == 1 ? 1 : 4098));
                CHECK(g.next(g.context) == 22177);
                ids[4098] = 22177;
                fail = 1;
                CHECK(g.materialize_step(g.context, ids, 4099, &cursor) == 0);
                g.work(g.context, &w);
                CHECK(w.decode_attempted == 1 && w.decode_committed == 0);
                fail = 0;
            }
            for (unsigned eos = 0; eos < 2; eos++) {
                g.reset(g.context);
                for (unsigned i = 0; i < 4353; i++) {
                    ids[i] = 1000 + (37 * i) % 8000;
                }
                g.begin(g.context);
                ingest(&g, ids, 4351, &w);
                if (eos) {
                    logits[2] = 2;
                }
                ids[4351] = (unsigned)g.next(g.context);
                CHECK(ids[4351] == (eos ? 2 : 22177));
                ingest(&g, ids, 4352, &w);
                CHECK(w.pending_committed == 1);
                CHECK(g.save(g.context) == 1);
                CHECK(g.restore(g.context) == 1);
                if (policy == 2) {
                    ingest(&g, ids, 4352, &w);
                    CHECK(w.recompute_committed == 4352);
                }
                CHECK(g.next(g.context) == -1); /* No room; never samples. */
            }
            g.reset(g.context);
            g.begin(g.context);
            CHECK(g.materialize_step(g.context, ids, 257, &cursor) == 2);
            CHECK(cursor == block);
            g.work(g.context, &w);
            fail = 1;
            CHECK(g.materialize_step(g.context, ids, 257, &cursor) == 0);
            CHECK(cursor == block);
            g.work(g.context, &w);
            CHECK(w.append_calls == 1 && w.suffix_attempted == (257 - block < block ? 257 - block : block));
            CHECK(!w.suffix_committed && !w.prefill_calls);
            fail = 0;
            /* Long byte admission uses the real additive tokenizer API once. */
            g.reset(g.context);
            unsigned char *text = malloc(32769);
            CHECK(text);
            memset(text, ' ', 32769);
            unsigned count = 4352;
            CHECK(g.frame(g.context, text, 32768, 1, ids, &count));
            CHECK(count == 515);
            count = 4352;
            CHECK(!g.frame(g.context, text, 32769, 1, ids, &count));
            free(text);
            g.destroy(g.context);
        }
    }
    puts("long native API/chunk/checkpoint/pending/failure/byte tests passed (no model)");
    return 0;
}
