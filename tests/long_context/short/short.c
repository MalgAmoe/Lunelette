/* Live observers and actual native/controller implementation, not a replica. */
#include "controller_probe.c"

static void short_run(native_service *n, int cancel)
{
    service_session *s = calloc(1, sizeof(*s));
    CHECK(s);
    service_generator g = {.context=n, .next=selected_next, .reset=native_reset,
        .destroy=retained_owner, .frame=native_frame, .piece=native_piece, .eos=2,
        .prefix_reuse=n->policy, .begin=native_begin, .save=counted_save,
        .restore=observed_restore, .invalidate=native_invalidate, .work=native_work,
        .materialize_step=gated_step, .context_capacity=4352, .block_capacity=n->block};
    probe_policy=n->policy;
    probe_block=n->block;
    CHECK(service_init(s,g));
    route(s,"/sessions","{}",200);
    route(s,"/sessions/1/reset","{}",200);
    probe_case="short_initial";
    if (cancel) {
        /* 126 digits + framing3 =129: pause after actual first128 forward. */
        digits(s,126,129);
        pthread_mutex_lock(&gate_lock);
        gate_target=1;
        gate_count=gate_held=gate_release=0;
        pthread_mutex_unlock(&gate_lock);
        route(s,"/sessions/1/generate","{\"id\":\"cancel-short\",\"budget\":1}",200);
        pthread_mutex_lock(&gate_lock);
        struct timespec deadline;
        clock_gettime(CLOCK_REALTIME,&deadline);
        deadline.tv_sec+=WAIT_SECONDS;
        while (!gate_held) {
            CHECK(!pthread_cond_timedwait(&gate_wake,&gate_lock,&deadline));
        }
        CHECK(gate_count==1 && n->length==128 && current_journal.tokens==128);
        pthread_mutex_unlock(&gate_lock);
        endpoint(n,"cancel_committed128_before_controller_receipt");
        route(s,"/requests/cancel-short/cancel","{}",200);
        pthread_mutex_lock(&gate_lock);
        gate_release=1;
        pthread_cond_broadcast(&gate_wake);
        pthread_mutex_unlock(&gate_lock);
        idle(s);
        CHECK(gate_count==1 && !sample_calls && !save_calls);
        CHECK(!strcmp(s->state,"needs_reset"));
        CHECK(!strcmp(s->history[s->active].status,"cancelled"));
        CHECK(!s->checkpoint_id && !n->checkpoint);
        gate_target=0;
        puts("{\"type\":\"short_cancel_verified\",\"prompt\":129,\"completed_chunk\":128,\"no_selection_or_save\":true}");
        route(s,"/sessions/1/reset","{}",200);
        digits(s,1,4);
        forced_id=9000;
        job(s,"generate",0);
    } else {
        digits(s,5,8);
        forced_id=2; /* Explicit forced EOS permits public checkpoint admission. */
        job(s,"generate",0);
        CHECK(s->logical==9 && s->pending==1);
        job(s,"checkpoint",0);
        CHECK(s->committed==9 && !s->pending && n->length==9);
        CHECK(s->history[s->active].work.pending_committed==1);
        CHECK(s->history[s->active].work.decode_calls==1);
        endpoint(n,"short_saved9");
        uint64_t saved=s->checkpoint_id;
        for (unsigned branch=0;branch<2;branch++) {
            digits(s,branch+1,branch+3);
            forced_id=9000+branch;
            job(s,"generate",0);
            CHECK(s->pending==1);
#ifdef CONTROLLER_STUB
            const char *fault=getenv("SHORT_FAULT");
            if (fault && !branch) {
                if (!strcmp(fault,"logits")) n->checkpoint->saved[131071]=NAN;
                else if (!strcmp(fault,"cursor")) n->saved_length++;
                else {
                    CHECK(!strcmp(fault,"input_state"));
                    n->runtime->ids[0]^=1; /* Stub has no KV; incompatible handle observation only. */
                }
            }
#endif
            job(s,"restore",saved);
            CHECK(s->logical==9 && !s->pending);
            if (n->policy==1) {
                CHECK(n->length==9 && s->committed==9);
            } else {
                CHECK(!n->length && !s->committed && !n->logits);
                /* Materialize restored ledger without replacing the saved identity. */
                native_begin(n);
                probe_case="short_OFF_rebuild9";
                service_work work={0};
                CHECK(materialize(s,&work)==1);
                CHECK(work.recompute_committed==9);
            }
        }
    }
    service_destroy(s); /* Join before independent reset/replay on same owner. */
    free(s);
    unsigned long long conservative_tokens=adapter_tokens;
    for (unsigned i=0;i<endpoint_count;i++) {
        conservative_tokens+=endpoint_records[i].journal.tokens;
    }
    CHECK(conservative_tokens<=(cancel ? 2048u : 1024u));
    direct_replays(n);
    puts("{\"type\":\"short_complete\",\"worker_joined_before_replay\":true}");
}

int main(int argc,char **argv)
{
    /* One configuration/owner per invocation; no implicit model matrix. */
    CHECK(argc==9);
    CHECK(!strcmp(argv[4],"mps"));
    CHECK(!strcmp(argv[5],"compact") || !strcmp(argv[5],"resident"));
    CHECK(!strcmp(argv[7],"on") || !strcmp(argv[7],"off"));
    int cancel=!strcmp(argv[8],"cancel");
    CHECK(cancel || !strcmp(argv[8],"normal"));
    CHECK(!strcmp(argv[3],cancel ? "128" : "256"));
    CHECK(strlen(argv[6])<=256);
    endpoint_dir=argv[6];
    unsigned policy=!strcmp(argv[7],"on") ? 1 : 2;
    native_service *n;
#ifdef CONTROLLER_STUB
    test_support(endpoint_dir);
    n=calloc(1,sizeof(*n));
    CHECK(n);
    n->runtime=calloc(1,sizeof(*n->runtime));
    CHECK(n->runtime);
    n->capacity=n->runtime->capacity=4352;
    n->block=n->runtime->block=(unsigned)atoi(argv[3]);
    n->policy=policy;
    char error[512];
    CHECK(lunelette_tokenizer_open(argv[2],&n->tokenizer,error,sizeof(error)));
#else
    @autoreleasepool {
    n=official_owner(policy,(unsigned)atoi(argv[3]),argv);
    CHECK(n);
#endif
    short_run(n,cancel);
    native_destroy(n);
#ifndef CONTROLLER_STUB
    }
#endif
    return 0;
}
