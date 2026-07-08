# SPDX-License-Identifier: Apache-2.0
"""独立 SSD 写进程 (Q2 sidecar): 把 decode 建的 SHM 环里的 KV block 落进 SSD.

职责单一 —— 只管 KV block 在 CPU(shm 环)与 SSD 之间的传输, 不碰 GPU、不碰
decode. 整个写路径 (账本 dedup / pwrite / fdatasync) 都在本进程 -> 它卡它自己,
和 decode 彻底隔离 (decode 侧只剩 capture 的 memmove, 实测 -1%).

启动 (launcher sidecar):
    python -m vllm.v1.core.sched.licht_v3.ssd_writer_process
读 env:
    LICHT_SSD_META_PATH  账本元数据目录 (shm)  —— decode 初始化后才有
    LICHT_SSD_PATH       SSD 数据文件目录
    LICHT_SSD_GB         冷库容量 (可选, 缺省从账本推)
    LICHT_SSD_RING_DIR   环所在目录 (默认 /dev/shm)
流程: 等账本 meta 就绪 -> open 账本 -> glob 发现各 decode 的环 -> 轮询 drain.
best-effort: 崩了 launcher 重拉; 环满 -> decode 侧丢 -> decode 不受影响.
"""
from __future__ import annotations

import glob
import logging
import os
import signal
import sys
import time

logger = logging.getLogger("licht.ssd_writer")

_RING_GLOB = "licht_ssd_stage_*.ring"
_stop = False


def _on_signal(signum, frame):
    global _stop
    _stop = True


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [ssd_writer] %(message)s")
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    meta = os.environ.get("LICHT_SSD_META_PATH")
    data = os.environ.get("LICHT_SSD_PATH")
    ring_dir = os.environ.get("LICHT_SSD_RING_DIR", "/dev/shm")
    if not meta or not data:
        logger.error("LICHT_SSD_META_PATH / LICHT_SSD_PATH 未设置 -> 退出")
        return 1

    from vllm.v1.core.sched.licht_v3.ssd_tier import SsdTier
    from vllm.v1.core.sched.licht_v3.ssd_stage_ring import StageRing

    # 1) 等账本 meta 就绪 (decode 初始化账本后写的)
    logger.info("等待账本 meta 就绪: %s", meta)
    m = None
    while not _stop:
        m = SsdTier.read_meta(meta)
        if m and m.get("slot_bytes"):
            break
        time.sleep(0.5)
    if _stop:
        return 0
    slot_bytes = int(m["slot_bytes"])
    block_size = int(m["block_size"])
    num_slots = int(m["num_slots"])
    ssd_gb = float(os.environ.get(
        "LICHT_SSD_GB", num_slots * slot_bytes / (1024 ** 3)))

    # 2) open 账本 (跨进程, 与 decode 同一个) —— 不 bind_cpu_source (无生产端/环)
    try:
        tier = SsdTier.open_or_create(
            meta_path=meta, data_path=data, ssd_gb=ssd_gb,
            slot_bytes=slot_bytes, block_size=block_size)
    except Exception as e:
        logger.error("open 账本失败: %s -> 退出", e)
        return 1
    logger.info("账本已开 (slot=%dB, block=%d). 轮询环目录 %s/%s",
                slot_bytes, block_size, ring_dir, _RING_GLOB)

    # 3) 每个环一个专属 drain 线程 (环是 SPSC, 单消费者). N 环 -> N 线程并发写
    #    -> pwrite 在账本锁外并发, 喂满盘 (~SATA 上限). 主线程只负责发现新环.
    import threading
    threads: dict[str, threading.Thread] = {}

    def _drain_loop(path: str, ring: "StageRing"):
        idle = 0
        while not _stop:
            try:
                n = tier.drain_ring(ring, max_items=1024)
            except Exception as e:   # pragma: no cover
                logger.warning("drain %s 出错: %s", path, e)
                n = 0
            if n == 0:
                # 环文件被删 (decode 退出) 且排空 -> 该线程收工
                if not os.path.exists(path) and ring.depth() == 0:
                    break
                idle = min(idle + 1, 20)
                time.sleep(0.002 * idle)
            else:
                idle = 0
        try:
            ring.close()
        except Exception:
            pass
        logger.info("环 %s drain 线程退出", path)

    while not _stop:
        try:
            for p in glob.glob(os.path.join(ring_dir, _RING_GLOB)):
                t = threads.get(p)
                if t is None or not t.is_alive():
                    try:
                        ring = StageRing.open(p)
                    except Exception:
                        continue   # 头还没就绪, 下轮再试
                    t = threading.Thread(target=_drain_loop, args=(p, ring),
                                         name=f"ssd-drain-{os.path.basename(p)}",
                                         daemon=True)
                    t.start()
                    threads[p] = t
                    logger.info("打开环 %s (专属 drain 线程)", p)
        except Exception:
            pass
        time.sleep(0.5)   # 发现新环的间隔 (drain 在各线程里持续跑)

    logger.info("收到停止信号, 退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
