#ifndef LUNELETTE_SERVICE_NATIVE_H
#define LUNELETTE_SERVICE_NATIVE_H
#include "session.h"
/* Startup on main thread; successful ownership transfers to the session worker.
 * No runtime calls on main after transfer. Caller destroys on init failure. */
int service_native_open(int argc, char **argv, service_generator *generator, unsigned *port);
#endif
