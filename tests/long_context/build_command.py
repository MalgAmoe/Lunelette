"""Synchronous POSIX build commands with an owned, non-escaping process group.

Only for the trusted compiler/Make recipes, which must not detach into new
sessions. Output goes to the caller's regular-file log, never a pipe. Descendant
zombies cannot be reaped by Python; cleanup requires no *live* group members.
"""
import os
import select
import signal
import subprocess
import time


def _process_census(*arguments):
    census = subprocess.Popen(['/bin/ps', *arguments],
                              stdout=subprocess.PIPE, text=True)
    try:
        output, _ = census.communicate(timeout=2)
    except BaseException:
        census.kill()
        census.communicate(timeout=2)
        raise
    if census.returncode:
        raise RuntimeError('build cleanup process census failed')
    return output


def _live_group(group):
    # Keep the leader unreaped until all group signalling is finished. Its PID
    # therefore cannot be recycled into an unrelated process/group during cleanup.
    output = _process_census('-axo', 'pgid=,stat=')
    return any(int(parts[0]) == group and not parts[1].startswith('Z')
               for line in output.splitlines() if len(parts := line.split()) == 2)


def run_build_command(argv, *, timeout, check=False, **kwargs):
    """Like run for file-backed build logs; cleanup precedes timeout/cancellation.

    Use a non-reaping exit notification, not Popen.poll/wait: retaining the
    leader prevents PID/PGID reuse even if it exits before its grandchildren.
    Cleanup is forceful: a 5 s polling budget, bounded 2 s census calls and
    a 5 s leader reap. Cleanup failure is explicit, never success.
    """
    cancellation = {signal.SIGINT, signal.SIGALRM}
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, cancellation)
    limits = kwargs.pop('preexec_fn', None)
    def child_setup():
        # Do not leak the parent's temporary cancellation mask into Make/tools.
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        if limits is not None:
            limits()
    try:
        process = subprocess.Popen(argv, start_new_session=True,
                                   preexec_fn=child_setup, **kwargs)
    except BaseException:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        raise
    events = None
    try:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        already_exited = False
        if hasattr(select, 'kqueue'):
            events = select.kqueue()
            try:
                events.control([select.kevent(process.pid, filter=select.KQ_FILTER_PROC,
                                             flags=select.KQ_EV_ADD,
                                             fflags=select.KQ_NOTE_EXIT)], 0, 0)
            except ProcessLookupError:
                # Darwin refuses NOTE_EXIT registration after a fast exit. Check
                # the retained child's zombie state without reaping its identity.
                already_exited = _process_census('-o', 'stat=', '-p',
                                                 str(process.pid)).strip().startswith('Z')
                if not already_exited:
                    raise
        deadline = time.monotonic() + timeout
        while not already_exited:
            if events is not None:
                exited = bool(events.control(None, 1, 0))
            else:
                exited = os.waitid(os.P_PID, process.pid,
                                   os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
            if exited:
                break
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(argv, timeout)
            time.sleep(min(.02, max(0, deadline-time.monotonic())))
    finally:
        # Also clean up a command that exits before its background children.
        # Defer repeated interactive cancellation until the group is quiescent.
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, cancellation)
        try:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                # Darwin can report EPERM for a group containing only zombies.
                # Never suppress a signalling failure while a live member remains.
                if _live_group(process.pid):
                    raise
            deadline = time.monotonic() + 5
            while _live_group(process.pid):
                if time.monotonic() >= deadline:
                    raise RuntimeError('build process group did not quiesce')
                time.sleep(.02)
        finally:
            try:
                process.wait(timeout=5)
            finally:
                try:
                    if events is not None:
                        events.close()
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, mask)
    result = subprocess.CompletedProcess(argv, process.returncode)
    if check:
        result.check_returncode()
    return result
