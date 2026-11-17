"""On-demand connections for aiosenseme (the SenseME/Haiku library Home Assistant uses).

Lives at custom_components/senseme/ondemand.py, imported by the integration's __init__.py
right after its Python 3.14 patch. Opt-in per fan via ON_DEMAND_ADDRESSES (empty = no change).
Write-up with the measurements: https://github.com/mikelawrence/senseme-hacs/issues/49

Why this exists (measured 2026-11-16): a first-gen Haiku keeps an eight-slot TCP table and
never times out a client that vanishes without a FIN. Its 802.11b radio drops often (31% frame retries at
-38 dBm), and every drop left the stock library's always-open socket behind as a zombie on
the fan: the library aborts after 90 quiet seconds (RST lost on a dead link) and reconnects.
Eight of those and the fan accepts-then-resets every new connection, which is the Sept 2026
"60 lost/restored lines a minute" symptom and a two-month outage nobody noticed.

What changes (for the fans listed in ON_DEMAND_ADDRESSES only; the rest keep the stock
library behaviour untouched): SensemeDevice no longer holds a socket open. Every `refresh_minutes` it opens a
connection, asks for a full update, reads until the fan goes quiet, and closes with a proper
FIN. Commands (speed, light, sleep mode...) do the same, batched over 50 ms, and read the
fan's echo so Home Assistant state updates at once. Availability now means "the last poll
succeeded"; it goes unavailable after FAILURES_BEFORE_UNAVAILABLE consecutive failures.

What is lost: push updates. The occupancy sensor and changes made from the remote show up
within one poll interval (default 1 min) instead of instantly.

Revert: delete the `ondemand.install()` line in the integration's __init__.py and restart.
"""

from __future__ import annotations

import asyncio
import logging
import random
import traceback

import aiosenseme.device as _device
from aiosenseme.device import SensemeDevice

_LOGGER = logging.getLogger(__name__)

# Only the fans listed here get on-demand connections; every other SenseME device keeps the
# stock always-open socket. Empty by default, so installing this module changes nothing until
# a fan's IP is added, e.g. {"192.168.1.23"}. A Haiku L on the same network never wedged in
# years on the stock code and its radio sleeps between packets (first byte of a reply takes
# ~0.8 s, sometimes 3+ s), so leave fans that are not misbehaving on the stock path.
ON_DEMAND_ADDRESSES: set[str] = set()

# A poll that fails this many times in a row marks the device unavailable. Three polls at the
# default one-minute rate is ~3 min, well under the offline alert's 30-minute window and long
# enough that one dropped Wi-Fi frame does not flap the entity.
FAILURES_BEFORE_UNAVAILABLE = 3
# The fan answers GETALL as a burst of ~40 small messages; a full second of silence AFTER the
# first byte means the reply is complete. The first byte itself gets longer, because a fan whose
# radio power-saves can take seconds to wake (measured 0.7-0.9 s typical, >3 s at times on the
# Haiku L). The hard cap bounds a session on a half-dead link.
FIRST_BYTE_SECONDS = 8.0
IDLE_SECONDS = 1.0
HARD_CAP_SECONDS = 15.0
CONNECT_TIMEOUT = 5.0
# Setters are synchronous properties; HA often sets two in one service call (e.g. power then
# speed). Collect them for this long and send them over one connection.
COMMAND_BATCH_SECONDS = 0.05


def _ensure_state(dev: SensemeDevice) -> None:
    """Attach per-device bookkeeping lazily, so patched instances need no new __init__."""
    if hasattr(dev, "_od_lock"):
        return
    dev._od_lock = asyncio.Lock()
    dev._od_pending: list[str] = []
    dev._od_flush_handle = None
    dev._od_failures = 0
    dev._od_poller_task = None
    dev._od_tasks: set[asyncio.Task] = set()


def _prefix(dev: SensemeDevice) -> str:
    # The stock library addresses the fan by MAC once known, by IP before that.
    return dev.mac or dev._address or "ALL"


async def _od_session(dev: SensemeDevice, commands: tuple[str, ...] = ()) -> bool:
    """One short-lived connection: send `commands`, request a full update, read until the fan
    goes quiet, close with FIN. Returns True when the fan answered."""
    _ensure_state(dev)
    async with dev._od_lock:
        writer = None
        ok = False
        should_callback = False
        loop = asyncio.get_running_loop()
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(dev._address, _device.PORT), CONNECT_TIMEOUT
            )
            prefix = _prefix(dev)
            for cmd in commands:
                writer.write(f"<{prefix};{cmd}>".encode("utf-8"))
            for cmd in ("DEVICE;ID;GET", "SNSROCC;STATUS;GET", "GETALL"):
                writer.write(f"<{prefix};{cmd}>".encode("utf-8"))
            await writer.drain()
            deadline = loop.time() + HARD_CAP_SECONDS
            leftover = ""
            got_any = False
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                wait = IDLE_SECONDS if got_any else FIRST_BYTE_SECONDS
                try:
                    chunk = await asyncio.wait_for(
                        reader.read(4096), timeout=min(wait, remaining)
                    )
                except asyncio.TimeoutError:
                    break  # quiet after data = reply complete; quiet before = no answer
                if not chunk:
                    break  # the fan closed first
                got_any = True
                leftover, cb = dev._process_message(
                    leftover + chunk.decode("utf-8", "replace")
                )
                should_callback = should_callback or cb
            ok = got_any
            if not ok:
                _LOGGER.debug("%s: connected to %s but got no reply", dev.name, dev.address)
        except (OSError, asyncio.TimeoutError):
            _LOGGER.debug(
                "%s: session to %s failed (%s)\n%s",
                dev.name, dev.address, "/".join(commands) or "poll", traceback.format_exc(),
            )
        finally:
            if writer is not None:
                writer.close()  # FIN, not RST: the fan must see us leave
                try:
                    await asyncio.wait_for(writer.wait_closed(), 3)
                except (OSError, asyncio.TimeoutError):
                    pass

        was_connected = dev._is_connected.is_set()
        if ok:
            dev._od_failures = 0
            dev._is_connected.set()
        else:
            dev._od_failures += 1
            if dev._od_failures >= FAILURES_BEFORE_UNAVAILABLE:
                dev._is_connected.clear()
        now_connected = dev._is_connected.is_set()
        if now_connected != was_connected:
            should_callback = True
            # Same wording the stock library logs, so existing log readers still match;
            # the very first connection after start is not a "restore", so it stays quiet.
            if not now_connected:
                dev._od_lost_logged = True
                _LOGGER.warning("%s: Connection to address %s lost", dev.name, dev.address)
            elif getattr(dev, "_od_lost_logged", False):
                dev._od_lost_logged = False
                _LOGGER.warning("%s: Connection to address %s restored", dev.name, dev.address)
        if should_callback:
            dev._execute_callbacks()
        return ok


_STOCK: dict = {}  # the library's original methods, captured by install()


def _on_demand(dev: SensemeDevice) -> bool:
    return dev._address in ON_DEMAND_ADDRESSES


def _od_send_command(dev: SensemeDevice, cmd: str) -> None:
    """Replacement for SensemeDevice._send_command: queue and flush in one session."""
    if not _on_demand(dev):
        return _STOCK["_send_command"](dev, cmd)
    _ensure_state(dev)
    dev._od_pending.append(cmd)
    if dev._od_flush_handle is None:
        loop = asyncio.get_running_loop()
        dev._od_flush_handle = loop.call_later(COMMAND_BATCH_SECONDS, _od_flush, dev)


def _od_flush(dev: SensemeDevice) -> None:
    dev._od_flush_handle = None
    cmds, dev._od_pending = tuple(dev._od_pending), []
    if not cmds:
        return
    task = asyncio.create_task(_od_session(dev, cmds))
    dev._od_tasks.add(task)
    task.add_done_callback(dev._od_tasks.discard)


async def _od_poller(dev: SensemeDevice) -> None:
    while True:
        try:
            await _od_session(dev)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the poller must outlive any surprise
            _LOGGER.error("%s: poll error\n%s", dev.name, traceback.format_exc())
        await asyncio.sleep(dev.refresh_minutes * 60 + random.uniform(-10, 10))


async def _od_async_update(
    dev: SensemeDevice, connection_lost: bool = False, timeout_seconds: int = 10
) -> bool:
    """Replacement for SensemeDevice.async_update.

    The stock version passes coroutines to asyncio.wait, which Python 3.12+ rejects (the HA
    fork shims asyncio to survive it), and it returns True even when the wait timed out. This
    one starts the poller, waits for the first successful poll, and reports honestly, so an
    unreachable fan at boot becomes ConfigEntryNotReady and HA retries on its own.
    """
    if not _on_demand(dev):
        return await _STOCK["async_update"](dev, connection_lost, timeout_seconds)
    if connection_lost:
        dev._connection_lost = True
    if not dev._is_running:
        dev.start()
    try:
        await asyncio.wait_for(
            asyncio.gather(dev._first_update.wait(), dev._is_connected.wait()),
            timeout_seconds,
        )
    except asyncio.TimeoutError:
        return False
    return True


def _od_start(dev: SensemeDevice) -> None:
    if not _on_demand(dev):
        return _STOCK["start"](dev)
    if dev._is_running:
        return
    _ensure_state(dev)
    dev._od_poller_task = asyncio.create_task(_od_poller(dev))
    dev._is_running = True
    _LOGGER.debug("%s: started (on-demand connections)", dev.name)


def _od_stop(dev: SensemeDevice) -> None:
    if not _on_demand(dev):
        return _STOCK["stop"](dev)
    if not dev._is_running:
        return
    _ensure_state(dev)
    if dev._od_poller_task is not None:
        dev._od_poller_task.cancel()
    if dev._od_flush_handle is not None:
        dev._od_flush_handle.cancel()
        dev._od_flush_handle = None
    for task in list(dev._od_tasks):
        task.cancel()
    dev._is_running = False


def install() -> None:
    """Patch SensemeDevice in place. Idempotent."""
    if getattr(SensemeDevice, "_od_installed", False):
        return
    for name in ("_send_command", "async_update", "start", "stop"):
        _STOCK[name] = getattr(SensemeDevice, name)
    SensemeDevice._send_command = _od_send_command
    SensemeDevice.async_update = _od_async_update
    SensemeDevice.start = _od_start
    SensemeDevice.stop = _od_stop
    SensemeDevice._od_installed = True
    _LOGGER.info(
        "aiosenseme patched: on-demand connections for %s", sorted(ON_DEMAND_ADDRESSES)
    )
