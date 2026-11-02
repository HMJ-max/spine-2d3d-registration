"""Small OS lock adapter. Lock lifetime follows the open file descriptor."""
import os
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8


def flock(file, operation):
    if os.name != 'nt':
        import fcntl
        return fcntl.flock(file, operation)
    import msvcrt
    fd = file if isinstance(file, int) else file.fileno()
    position = os.lseek(fd, 0, os.SEEK_CUR)
    os.lseek(fd, 0, os.SEEK_SET)
    mode = msvcrt.LK_UNLCK if operation & LOCK_UN else msvcrt.LK_NBLCK if operation & LOCK_NB else msvcrt.LK_LOCK
    try:
        msvcrt.locking(fd, mode, 1)
    finally:
        os.lseek(fd, position, os.SEEK_SET)
