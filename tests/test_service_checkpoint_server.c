/* CPU-only native HTTP integration: no Metal runtime or model weights. */
#define _POSIX_C_SOURCE 200809L
#include "support/native_runtime.h"
#include <time.h>
#include <errno.h>

/* Delay controls only, leaving the actual adapter/controller/transport intact.
 * Polling must observe an admitted control before explicit cancellation. */
static int delayed_save(void *context);
static int delayed_restore(void *context);
static int (*adapter_save)(void *);
static int (*adapter_restore)(void *);
static void control_delay(void)
{
    struct timespec delay = {0, 100000000};
    while (nanosleep(&delay, &delay) && errno == EINTR) {
    }
}
static int delayed_save(void *context)
{
    control_delay();
    return adapter_save(context);
}
static int delayed_restore(void *context)
{
    control_delay();
    return adapter_restore(context);
}
static int stub_native_open(int argc, char **argv, service_generator *g, unsigned *port)
{
    if (!service_native_open(argc, argv, g, port)) {
        return 0;
    }
    adapter_save = g->save;
    adapter_restore = g->restore;
    g->save = delayed_save;
    g->restore = delayed_restore;
    return 1;
}
#define SERVICE_NATIVE
#define service_native_open stub_native_open
#include "../service/http.c"
#undef service_native_open
