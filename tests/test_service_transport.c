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
    /* Closing a terminal/old stream cannot target a newly admitted job. */
    service_session *session = calloc(1, sizeof(*session));
    if (!session || pthread_mutex_init(&session->lock, NULL)) {
        return 1;
    }
    int sockets[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets)) {
        return 1;
    }
    session->job = 1;
    strcpy(session->history[0].id, "new-control");
    strcpy(session->history[0].operation, "checkpoint");
    connection old = {.fd = sockets[0], .streaming = 1};
    strcpy(old.request_id, "old-generation");
    close_connection(&old, session);
    close(sockets[1]);
    if (session->cancel) {
        return 1;
    }
    strcpy(session->history[0].operation, "generate");
    strcpy(session->history[0].id, "new-generation");
    service_disconnect_request(session, "old-generation");
    if (session->cancel) {
        return 1;
    }
    service_disconnect_request(session, "new-generation");
    if (!session->cancel || strcmp(session->history[0].cancel_source, "disconnect")) {
        return 1;
    }
    pthread_mutex_destroy(&session->lock);
    free(session);
    char *arguments[] = {"test_service_server", "--port", "0", NULL};
    int status = service_http_main(3, arguments);
    if (status != 1 || poll_calls != 2) {
        fprintf(stderr, "poll failure regression: status=%d calls=%u\n", status, poll_calls);
        return 1;
    }
    puts("service transport EINTR/fatal-poll cleanup passed");
    return 0;
}
