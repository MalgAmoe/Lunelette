#ifndef LUNELETTE_SERVICE_NATIVE_H
#define LUNELETTE_SERVICE_NATIVE_H
#include "session.h"
/* Startup on main thread; successful ownership transfers to the session worker.
 * No runtime calls on main after transfer. Caller destroys on init failure.
 * next/replay_next return token (including EOS 2) or -1 without publishing a new
 * emitted/pending token. Selection borrows immutable runtime logits through
 * capture: worker ownership excludes reset/materialize/restore in that interval.
 * A capture failure may write partial evidence, latches replay failure, and
 * leaves capture index unchanged; reset does not clear this latch/index. */
int service_native_open(int argc, char **argv, service_generator *generator, unsigned *port);
#endif
