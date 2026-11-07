"""Local Unix-socket control channel. No DDS or motor access."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import stat
import tempfile
import time


class ControlSocket:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.sock = None
        self.lockfd = None
        self.closed = False
        if not self.path.parent.is_dir() or self.path.parent.stat().st_uid != os.getuid():
            raise ValueError('control socket parent must be an existing directory owned by this user')
        self.lockfd = os.open(str(self.path) + '.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.is_symlink():
                raise ValueError('refusing a symlink at the control socket path')
            if self.path.exists():
                info = self.path.lstat()
                if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
                    raise ValueError('refusing to replace a non-socket or another user\'s socket')
                self.path.unlink()
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self.sock.bind(str(self.path))
            os.chmod(self.path, 0o600)
            self.sock.setblocking(False)
        except BaseException:
            if self.sock:
                self.sock.close()
            os.close(self.lockfd)
            self.lockfd = None
            raise

    def poll(self):
        try:
            data, address = self.sock.recvfrom(4097)
        except BlockingIOError:
            return None
        try:
            if len(data) > 4096:
                raise ValueError('control packet too large')
            payload = json.loads(data)
            if not isinstance(payload, dict):
                raise ValueError('expected a JSON object')
            sent = payload.get('sent')
            if not isinstance(sent, (int, float)) or not 0 <= time.monotonic() - sent <= 1:
                raise ValueError('control request expired; commands are never queued for a later state')
            if payload.get('command') not in ('status', 'select'):
                raise ValueError('unknown control command')
            if payload['command'] == 'select' and not isinstance(payload.get('motion'), str):
                raise ValueError('select requires a motion name')
            return payload, address
        except (ValueError, TypeError) as exc:
            self.reply(address, {'ok': False, 'message': str(exc)})
            return None

    def reply(self, address, payload):
        if not address:
            return
        try:
            self.sock.sendto(json.dumps(payload).encode(), address)
        except OSError:
            pass  # A disconnected control client never affects balance.

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.sock:
            self.sock.close()
            self.path.unlink(missing_ok=True)
        if self.lockfd is not None:
            os.close(self.lockfd)


def request(path: Path, command: str, motion: str | None = None):
    with tempfile.TemporaryDirectory(prefix='supermimic-control-') as directory:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
            client.bind(str(Path(directory) / 'client.sock'))
            client.settimeout(2)
            client.sendto(json.dumps({'command': command, 'motion': motion, 'sent': time.monotonic()}).encode(), str(path))
            return json.loads(client.recv(65536))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('motion', help='Motion alias to select, or status')
    parser.add_argument('--socket', required=True, type=Path)
    args = parser.parse_args()
    try:
        response = request(args.socket, 'status' if args.motion == 'status' else 'select', args.motion)
    except OSError as exc:
        parser.exit(1, f'Control connection failed: {exc}. Start the live runner first.\n')
    print(json.dumps(response, indent=2))
    if not response.get('ok'):
        parser.exit(1)


if __name__ == '__main__':
    main()
