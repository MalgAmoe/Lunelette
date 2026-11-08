#ifndef LUNELETTE_TEST_SUPPORT_SESSION_FIXTURE_H
#define LUNELETTE_TEST_SUPPORT_SESSION_FIXTURE_H

/* Header-local deterministic callbacks and state shared by session/replay tests.
 * Define _POSIX_C_SOURCE before system headers. Scenario assertions stay in callers. */

#define _POSIX_C_SOURCE 200809L
#include <limits.h>
#include "session.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
typedef struct {
    unsigned position, length, prefix[256], resets, destroys;
    int invalid;
    pthread_mutex_t lock;
    pthread_cond_t wake;
    unsigned block_at;
    int entered, released;
    unsigned saved_length, saved_position, saved_prefix[256];
    int save_failure, restore_failure, block_save, block_restore, commit_failure;
    unsigned policy;
    pthread_t callback_thread;
} fake;
static int next(void *context)
{
    fake *f = context;
    pthread_mutex_lock(&f->lock);
    if (f->block_at && f->position == f->block_at) {
        f->entered = 1;
        pthread_cond_broadcast(&f->wake);
        while (!f->released) {
            pthread_cond_wait(&f->wake, &f->lock);
        }
    }
    pthread_mutex_unlock(&f->lock);
    static const unsigned pieces[] = {0xe2, 0x82, 0xac, 0xf0, 0x9f, 0x98, 0x80, 256};
    if (f->invalid == 2) {
        return 'x';
    }
    return f->invalid ? 0xff : (int)pieces[f->position++ % 8];
}
static int commit(void *context, const unsigned *tokens, unsigned length)
{
    fake *f = context;
    CHECK(pthread_equal(pthread_self(), f->callback_thread));
    if (f->commit_failure) {
        return 0;
    }
    CHECK(length >= f->length);
    CHECK(!memcmp(tokens, f->prefix, f->length * sizeof(*tokens)));
    memcpy(f->prefix, tokens, length * sizeof(*tokens));
    f->length = length;
    return 1;
}
static void reset(void *context)
{
    fake *f = context;
    f->callback_thread = pthread_self();
    f->saved_length = 0;
    f->position = f->length = 0;
    f->resets++;
}
static void destroy(void *context)
{
    ((fake *)context)->destroys++;
}
static service_session session;
static char reply[4096];
static int route(const char *method, const char *path, const char *body)
{
    int stream;
    return service_route(&session, method, path, body, strlen(body), reply, sizeof(reply), &stream);
}

#endif
