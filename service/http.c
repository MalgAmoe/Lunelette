#define _POSIX_C_SOURCE 200809L
#include "session.h"
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define CLIENTS 16
#define HEADER_LIMIT 4096
#define BODY_LIMIT 8192
#define INPUT_LIMIT (HEADER_LIMIT + BODY_LIMIT)
typedef struct {
    int fd, streaming, closing, parsed;
    size_t received, header, body, sent, output_length;
    double deadline;
    char input[INPUT_LIMIT + 1], output[8192];
} connection;
static volatile sig_atomic_t stopping;
static void signal_stop(int number)
{
    (void)number;
    stopping = 1;
}
static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + t.tv_nsec / 1e9;
}
static int nonblocking(int fd)
{
    int flags = fcntl(fd, F_GETFL, 0);
    return flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}
static void close_connection(connection *c, service_session *s)
{
    if (c->streaming) {
        service_disconnect(s);
    }
    close(c->fd);
    memset(c, 0, sizeof(*c));
    c->fd = -1;
}
static void response(connection *c, int status, const char *body, int streaming)
{
    c->streaming = streaming;
    c->closing = !streaming;
    c->sent = 0;
    if (streaming) {
        c->output_length = (size_t)snprintf(c->output, sizeof(c->output),
            "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nConnection: close\r\n\r\n");
    } else {
        c->output_length = (size_t)snprintf(c->output, sizeof(c->output),
            "HTTP/1.1 %d Result\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n"
            "Connection: close\r\n\r\n%s", status, strlen(body), body);
    }
    c->deadline = now() + 2.0;
}
static void bad_request(service_session *session, connection *c)
{
    char reply[512];
    service_transport_error(session, "invalid_http", reply, sizeof(reply));
    response(c, 400, reply, 0);
}
/* Parse only complete bounded headers. Content-Length is required for every
 * request, including zero-length GET. Reject transfer coding and pipelining. */
static int headers(connection *c, char *method, char *path)
{
    char *end = strstr(c->input, "\r\n\r\n");
    if (!end) {
        return c->received >= HEADER_LIMIT ? -1 : 0;
    }
    c->header = (size_t)(end - c->input) + 4;
    if (c->header > HEADER_LIMIT || memchr(c->input, 0, c->received)) {
        return -1;
    }
    char copy[HEADER_LIMIT + 1];
    memcpy(copy, c->input, c->header);
    copy[c->header] = 0;
    char *line = strstr(copy, "\r\n");
    if (!line) {
        return -1;
    }
    *line = 0;
    int used = 0;
    if (sscanf(copy, "%15s %255s HTTP/1.1%n", method, path, &used) != 2 ||
        !used || copy[used] || path[0] != '/') {
        return -1;
    }
    unsigned lengths = 0, hosts = 0;
    char *p = line + 2;
    while (*p != '\r') {
        char *next = strstr(p, "\r\n");
        if (!next || *p == ' ' || *p == '\t') {
            return -1;
        }
        *next = 0;
        char *colon = strchr(p, ':');
        if (!colon || colon == p) {
            return -1;
        }
        for (char *q = p; q < colon; q++) {
            if (!( (*q >= 'a' && *q <= 'z') || (*q >= 'A' && *q <= 'Z') ||
                   (*q >= '0' && *q <= '9') || *q == '-')) {
                return -1;
            }
        }
        *colon++ = 0;
        while (*colon == ' ' || *colon == '\t') {
            colon++;
        }
        if (!strcasecmp(p, "Transfer-Encoding")) {
            return -1;
        }
        if (!strcasecmp(p, "Host")) {
            if (++hosts != 1 || !*colon) {
                return -1;
            }
        }
        if (!strcasecmp(p, "Content-Length")) {
            if (++lengths != 1 || !*colon) {
                return -1;
            }
            size_t length = 0;
            for (; *colon >= '0' && *colon <= '9'; colon++) {
                length = length * 10 + (unsigned)(*colon - '0');
                if (length > BODY_LIMIT) {
                    return -1;
                }
            }
            while (*colon == ' ' || *colon == '\t') {
                colon++;
            }
            if (*colon) {
                return -1;
            }
            c->body = length;
        }
        p = next + 2;
    }
    return hosts == 1 && lengths == 1 ? 1 : -1;
}

/* Test generator: fixed raw pieces split every byte of A€😀Z, then EOS.
 * materialized prefix is checked byte-for-byte on every append. */
typedef struct {
    unsigned prefix[256], length, position, delay_ms, emitted;
    int has_pending;
} fixture_generator;
static int fixture_materialize(void *context, const unsigned *tokens, unsigned length)
{
    fixture_generator *f = context;
    if (length < f->length || memcmp(f->prefix, tokens, f->length * sizeof(*tokens))) {
        return 0;
    }
    unsigned appended = f->length;
    if (f->has_pending) {
        if (length <= appended || tokens[appended] != f->emitted ||
            (f->emitted != 256 && length != appended + 1)) {
            return 0;
        }
        appended++;
    }
    for (unsigned i = appended; i < length; i++) {
        if (tokens[i] > 255) {
            return 0;
        }
    }
    memcpy(f->prefix, tokens, length * sizeof(*tokens));
    f->length = length;
    f->has_pending = 0;
    return 1;
}
static int fixture_next(void *context)
{
    fixture_generator *f = context;
    static const unsigned row[] = {'A', 0xe2, 0x82, 0xac, 0xf0, 0x9f, 0x98, 0x80, 'Z', 256};
    struct timespec delay = {0, (long)f->delay_ms * 1000000};
    if (f->delay_ms == 1000) {
        delay.tv_sec = 1;
        delay.tv_nsec = 0;
    }
    while (nanosleep(&delay, &delay) && errno == EINTR) {}
    if (f->has_pending) {
        return -1;
    }
    f->emitted = row[f->position++ % (sizeof(row) / sizeof(row[0]))];
    f->has_pending = 1;
    return (int)f->emitted;
}
static void fixture_reset(void *context)
{
    fixture_generator *f = context;
    f->length = f->position = 0;
    f->has_pending = 0;
}
static void fixture_destroy(void *context)
{
    (void)context;
}
int main(int argc, char **argv)
{
    unsigned port = 0, delay = 10;
    int port_seen = 0;
    for (int i = 1; i < argc; i += 2) {
        char *end;
        if (i + 1 >= argc || !*argv[i + 1] || argv[i + 1][0] == '-') {
            return 2;
        }
        errno = 0;
        unsigned long value = strtoul(argv[i + 1], &end, 10);
        if (errno || *end) {
            return 2;
        }
        if (!strcmp(argv[i], "--port") && value <= 65535) {
            port = (unsigned)value;
            port_seen = 1;
        } else if (!strcmp(argv[i], "--tick-ms") && value <= 1000) {
            delay = (unsigned)value;
        } else {
            return 2;
        }
    }
    if (!port_seen) {
        fprintf(stderr, "test executable requires --port (0 selects ephemeral loopback port)\n");
        return 2;
    }
    struct sigaction action = {0};
    action.sa_handler = signal_stop;
    sigemptyset(&action.sa_mask);
    sigaction(SIGTERM, &action, NULL);
    sigaction(SIGINT, &action, NULL);
    signal(SIGPIPE, SIG_IGN);
    int listener = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in address = {0};
    address.sin_family = AF_INET;
    inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
    address.sin_port = htons((unsigned short)port);
    if (listener < 0 || !nonblocking(listener) ||
        bind(listener, (struct sockaddr *)&address, sizeof(address)) || listen(listener, CLIENTS)) {
        perror("loopback listen");
        if (listener >= 0) {
            close(listener);
        }
        return 1;
    }
    socklen_t address_length = sizeof(address);
    if (getsockname(listener, (struct sockaddr *)&address, &address_length)) {
        close(listener);
        return 1;
    }
    fixture_generator fixture = {.delay_ms = delay};
    service_generator generator = {&fixture, fixture_next, fixture_materialize,
                                   fixture_reset, fixture_destroy};
    service_session *session = calloc(1, sizeof(*session));
    if (!session || !service_init(session, generator)) {
        free(session);
        close(listener);
        return 1;
    }
    fprintf(stderr, "{\"event\":\"listening\",\"port\":%u,\"model_loaded\":false}\n",
            ntohs(address.sin_port));
    fflush(stderr);
    connection clients[CLIENTS] = {0};
    for (unsigned i = 0; i < CLIENTS; i++) {
        clients[i].fd = -1;
    }
    int stream_owner = -1;
    int exit_status = 0;
    while (!stopping) {
        struct pollfd polls[CLIENTS + 1] = {{listener, POLLIN, 0}};
        for (unsigned i = 0; i < CLIENTS; i++) {
            connection *c = &clients[i];
            if (c->fd >= 0 && c->streaming && c->sent == c->output_length) {
                c->sent = c->output_length = 0;
                if (service_event(session, c->output)) {
                    c->output_length = strlen(c->output);
                    if (strstr(c->output, "\"event\":\"completed\"") ||
                        strstr(c->output, "\"event\":\"cancelled\"") ||
                        strstr(c->output, "\"event\":\"error\"")) {
                        c->closing = 1;
                    }
                }
            }
            polls[i + 1].fd = c->fd;
            polls[i + 1].events = POLLIN;
            if (c->output_length > c->sent) {
                polls[i + 1].events |= POLLOUT;
            }
        }
        int result = poll(polls, CLIENTS + 1, 50);
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            perror("service poll");
            exit_status = 1;
            break;
        }
        if (polls[0].revents & POLLIN) {
            int fd = accept(listener, NULL, NULL);
            if (fd >= 0) {
                unsigned slot;
                for (slot = 0; slot < CLIENTS && clients[slot].fd >= 0; slot++) {}
                if (slot == CLIENTS || !nonblocking(fd)) {
                    close(fd);
                } else {
                    clients[slot].fd = fd;
                    clients[slot].deadline = now() + 2.0;
                }
            }
        }
        for (unsigned i = 0; i < CLIENTS; i++) {
            connection *c = &clients[i];
            if (c->fd < 0) {
                continue;
            }
            int revents = polls[i + 1].revents;
            int dead = (revents & (POLLERR | POLLHUP | POLLNVAL)) != 0;
            if (!dead && (revents & POLLIN)) {
                ssize_t n = recv(c->fd, c->input + c->received,
                                 INPUT_LIMIT - c->received, 0);
                if (n <= 0) {
                    dead = n == 0 || (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR);
                } else if (c->parsed || c->streaming) {
                    dead = 1;
                } else {
                    c->received += (size_t)n;
                    c->input[c->received] = 0;
                    char method[16], path[256];
                    int parsed = headers(c, method, path);
                    if (parsed < 0) {
                        c->parsed = 1;
                        bad_request(session, c);
                    } else if (parsed > 0 && c->received >= c->header + c->body) {
                        c->parsed = 1;
                        if (c->received != c->header + c->body) {
                            bad_request(session, c);
                        } else {
                            char reply[4096];
                            int stream;
                            /* A completed stream must finish transport delivery before
                             * another generation can consume the shared event ring. */
                            int status;
                            if (stream_owner >= 0 && strstr(path, "/generate")) {
                                status = 409;
                                stream = 0;
                                service_transport_error(session, "busy_stream", reply, sizeof(reply));
                            } else {
                                status = service_route(session, method, path,
                                    c->input + c->header, c->body, reply, sizeof(reply), &stream);
                            }
                            response(c, status, reply, stream);
                            if (stream) {
                                stream_owner = (int)i;
                            }
                        }
                    }
                }
            }
            if (!dead && (revents & POLLOUT) && c->sent < c->output_length) {
                ssize_t n = send(c->fd, c->output + c->sent, c->output_length - c->sent, 0);
                if (n > 0) {
                    c->sent += (size_t)n;
                    c->deadline = now() + 2.0;
                } else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
                    dead = 1;
                }
            }
            if (now() >= c->deadline ||
                (c->closing && c->sent == c->output_length)) {
                dead = 1;
            }
            if (dead) {
                close_connection(c, session);
                if (stream_owner == (int)i) {
                    stream_owner = -1;
                }
            }
        }
    }
    close(listener);
    for (unsigned i = 0; i < CLIENTS; i++) {
        if (clients[i].fd >= 0) {
            close_connection(&clients[i], session);
        }
    }
    service_destroy(session);
    free(session);
    return exit_status;
}
