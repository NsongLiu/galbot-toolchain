#!/usr/bin/env python3
"""模型权重文件传输工具（TCP/UDP 双端 + SHA256 完整性校验）

用法:
  # 服务端（发送方）
  python transfer_weights.py server model.safetensors --proto tcp --port 9999
  python transfer_weights.py server model.safetensors --proto udp --port 9999

  # 客户端（接收方）
  python transfer_weights.py client 127.0.0.1 --proto tcp --port 9999 -o ./recv
  python transfer_weights.py client 127.0.0.1 --proto udp --port 9999 -o ./recv
"""

import argparse
import hashlib
import json
import os
import socket
import struct
import sys
import time

BUF = 1 << 16          # 读文件/socket 缓冲 64KB
UDP_CHUNK = 16384      # UDP 单包载荷。注意: chunk*window 为在途字节, 超过内核 UDP
                       # 缓冲(本机约 200KB 有效)会大量丢包导致吞吐崩塌
UDP_WINDOW = 8         # 滑动窗口大小（Go-Back-N）
UDP_TIMEOUT = 0.1      # 超时重传（秒）；UDP 丢包难免，超时要小以快速恢复

PKT_META = b"M"        # 元信息包
PKT_DATA = b"D"        # 数据包:  b"D" + seq(8B) + payload
PKT_FIN = b"F"         # 结束包:  b"F" + seq(8B, 总包数)
PKT_ACK = b"A"         # 累积ACK: b"A" + next_expected(8B)
PKT_FACK = b"K"        # FIN 的 ACK
PKT_GET = b"G"         # 客户端请求开始


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(BUF)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_meta(path):
    return {
        "name": os.path.basename(path),
        "size": os.path.getsize(path),
        "sha256": sha256_file(path),
    }


def verify(path, meta):
    size_ok = os.path.getsize(path) == meta["size"]
    hash_now = sha256_file(path)
    hash_ok = hash_now == meta["sha256"]
    return size_ok, hash_ok, hash_now


def fmt_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024


# ---------------------------------------------------------------- TCP
def tcp_serve(path, host, port):
    meta = file_meta(path)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[TCP] 监听 {host}:{port}, 文件: {meta['name']} ({fmt_size(meta['size'])})")
    while True:
        conn, addr = srv.accept()
        print(f"[TCP] 客户端接入: {addr}")
        try:
            header = json.dumps(meta).encode()
            conn.sendall(struct.pack("!I", len(header)) + header)
            t0 = time.time()
            with open(path, "rb") as f:
                while True:
                    b = f.read(BUF)
                    if not b:
                        break
                    conn.sendall(b)
            dt = time.time() - t0
            print(f"[TCP] 发送完成, 耗时 {dt:.2f}s ({fmt_size(meta['size']/max(dt,1e-6))}/s)")
        except (BrokenPipeError, ConnectionResetError):
            print("[TCP] 连接中断")
        finally:
            conn.close()


def tcp_fetch(host, port, outdir):
    sock = socket.create_connection((host, port))
    (raw_len,) = struct.unpack("!I", _recv_exact(sock, 4))
    meta = json.loads(_recv_exact(sock, raw_len))
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, meta["name"])
    print(f"[TCP] 接收 {meta['name']} ({fmt_size(meta['size'])}) -> {out}")
    t0 = time.time()
    left = meta["size"]
    with open(out + ".part", "wb") as f:
        while left > 0:
            b = sock.recv(min(BUF, left))
            if not b:
                raise ConnectionError("连接提前断开, 数据不完整")
            f.write(b)
            left -= len(b)
    dt = time.time() - t0
    sock.close()
    _finish(out, meta, dt)


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        b = sock.recv(n - len(buf))
        if not b:
            raise ConnectionError("连接提前断开")
        buf += b
    return buf


# ---------------------------------------------------------------- UDP (Go-Back-N)
def _set_bufs(sock):
    for opt in (socket.SO_RCVBUF, socket.SO_SNDBUF):
        sock.setsockopt(socket.SOL_SOCKET, opt, 4 << 20)


def udp_serve(path, host, port):
    meta = file_meta(path)
    total = (meta["size"] + UDP_CHUNK - 1) // UDP_CHUNK
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _set_bufs(sock)
    sock.bind((host, port))
    sock.settimeout(None)
    print(f"[UDP] 监听 {host}:{port}, 文件: {meta['name']} ({fmt_size(meta['size'])}, {total} 包)")

    while True:
        data, addr = sock.recvfrom(2048)          # 等待客户端 GET
        if data != PKT_GET:
            continue
        print(f"[UDP] 客户端接入: {addr}")
        t0 = time.time()
        # 先发元信息（重发直至收到第一个数据 ACK）
        meta_pkt = PKT_META + json.dumps(meta).encode()
        sock.settimeout(UDP_TIMEOUT)
        while True:                                # 收到任意 ACK 说明客户端已拿到元信息
            sock.sendto(meta_pkt, addr)
            try:
                ack, _ = sock.recvfrom(2048)
                if ack[:1] == PKT_ACK:
                    break
            except socket.timeout:
                pass
        base = 0
        # Go-Back-N 滑窗发送
        nxt = base
        with open(path, "rb") as f:
            cache = {}
            while base < total:
                while nxt < base + UDP_WINDOW and nxt < total:
                    f.seek(nxt * UDP_CHUNK)
                    payload = f.read(UDP_CHUNK)
                    pkt = PKT_DATA + struct.pack("!Q", nxt) + payload
                    cache[nxt] = pkt
                    sock.sendto(pkt, addr)
                    nxt += 1
                try:
                    while True:                   # 一次性排空已到达的 ACK
                        ack, _ = sock.recvfrom(2048)
                        if ack[:1] == PKT_ACK:
                            new_base = struct.unpack("!Q", ack[1:9])[0]
                            for i in range(base, min(new_base, nxt)):
                                cache.pop(i, None)
                            base = max(base, new_base)
                        sock.setblocking(False)
                except BlockingIOError:
                    sock.setblocking(True)
                    sock.settimeout(UDP_TIMEOUT)
                except socket.timeout:
                    for i in range(base, nxt):      # 超时: 从 base 起全部重发
                        sock.sendto(cache[i], addr)
        # 发 FIN, 等 FACK
        fin = PKT_FIN + struct.pack("!Q", total)
        while True:
            sock.sendto(fin, addr)
            try:
                ack, _ = sock.recvfrom(2048)
                if ack == PKT_FACK:
                    break
            except socket.timeout:
                pass
        dt = time.time() - t0
        print(f"[UDP] 发送完成, 耗时 {dt:.2f}s ({fmt_size(meta['size']/max(dt,1e-6))}/s)")
        sock.settimeout(None)


def udp_fetch(host, port, outdir):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _set_bufs(sock)
    sock.settimeout(10.0)
    # 发 GET, 收元信息
    meta = None
    while meta is None:
        sock.sendto(PKT_GET, (host, port))
        try:
            data, _ = sock.recvfrom(65535)
            if data[:1] == PKT_META:
                meta = json.loads(data[1:])
        except socket.timeout:
            print("[UDP] 等待服务端响应超时, 重试...")
    total = (meta["size"] + UDP_CHUNK - 1) // UDP_CHUNK
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, meta["name"])
    print(f"[UDP] 接收 {meta['name']} ({fmt_size(meta['size'])}, {total} 包) -> {out}")

    t0 = time.time()
    expected = 0
    sock.settimeout(30.0)
    # GBN 保证按序到达, 直接顺序写缓冲流即可
    with open(out + ".part", "wb", buffering=1 << 20) as f:
        sock.sendto(PKT_ACK + struct.pack("!Q", 0), (host, port))  # 通知服务端开始发数据
        while True:
            data, _ = sock.recvfrom(65535)
            tag = data[:1]
            if tag == PKT_DATA:
                (seq,) = struct.unpack("!Q", data[1:9])
                if seq == expected:               # GBN 接收端: 只收按序包, 其余丢弃
                    f.write(data[9:])
                    expected += 1
                sock.sendto(PKT_ACK + struct.pack("!Q", expected), (host, port))
            elif tag == PKT_META:                 # 元信息重传(ACK丢失): 回复当前进度
                sock.sendto(PKT_ACK + struct.pack("!Q", expected), (host, port))
            elif tag == PKT_FIN:
                (fin_total,) = struct.unpack("!Q", data[1:9])
                if fin_total == total and expected == total:
                    sock.sendto(PKT_FACK, (host, port))
                    break
                sock.sendto(PKT_ACK + struct.pack("!Q", expected), (host, port))
    dt = time.time() - t0
    sock.close()
    _finish(out, meta, dt)


# ---------------------------------------------------------------- 公共
def _finish(out, meta, dt):
    size_ok, hash_ok, hash_now = verify(out + ".part", meta)
    print(f"[校验] 大小: {'OK' if size_ok else 'MISMATCH'}  "
          f"SHA256: {'OK' if hash_ok else 'MISMATCH'}  ({hash_now[:16]}...)")
    if size_ok and hash_ok:
        os.replace(out + ".part", out)
        print(f"[完成] {out}  耗时 {dt:.2f}s ({fmt_size(meta['size']/max(dt,1e-6))}/s)")
    else:
        print(f"[失败] 校验不通过, 保留不完整文件: {out}.part", file=sys.stderr)
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description="模型权重文件传输 (TCP/UDP + SHA256 校验)")
    sub = p.add_subparsers(dest="role", required=True)

    ps = sub.add_parser("server", help="发送方")
    ps.add_argument("file", help="要发送的权重文件")
    ps.add_argument("--proto", choices=["tcp", "udp"], default="tcp")
    ps.add_argument("--host", default="0.0.0.0")
    ps.add_argument("--port", type=int, default=9999)

    pc = sub.add_parser("client", help="接收方")
    pc.add_argument("host", help="服务端地址")
    pc.add_argument("--proto", choices=["tcp", "udp"], default="tcp")
    pc.add_argument("--port", type=int, default=9999)
    pc.add_argument("-o", "--outdir", default="./recv_weights")

    args = p.parse_args()
    if args.role == "server":
        if not os.path.isfile(args.file):
            sys.exit(f"文件不存在: {args.file}")
        (tcp_serve if args.proto == "tcp" else udp_serve)(args.file, args.host, args.port)
    else:
        (tcp_fetch if args.proto == "tcp" else udp_fetch)(args.host, args.port, args.outdir)


if __name__ == "__main__":
    main()
