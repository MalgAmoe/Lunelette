/* Test-only syscall failure injection. The maintained transport is unchanged by
 * these macros; optional service-test already requires loopback sockets. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned poll_calls;
static int injected_poll(struct pollfd *fds, nfds_t count, int timeout)
{
    (void)fds;
    (void)count;
    (void)timeout;
    errno = ++poll_calls == 1 ? EINTR : EINVAL;
    return -1;
}

#define poll injected_poll
#define main service_http_main
#include "../service/http.c"
#undef main
#undef poll

int main(void)
{
    char *arguments[] = {"test_service_server", "--port", "0", NULL};
    int status = service_http_main(3, arguments);
    if (status != 1 || poll_calls != 2) {
        fprintf(stderr, "poll failure regression: status=%d calls=%u\n", status, poll_calls);
        return 1;
    }
    puts("service transport EINTR/fatal-poll cleanup passed");
    return 0;
}
