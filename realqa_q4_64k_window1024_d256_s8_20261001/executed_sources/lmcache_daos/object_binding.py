"""ctypes binding for the DFS-bypass DAOS GPU object shim.

The shim stores every LMCache entry below one fixed DAOS object:

* dkey: the LMCache cache-key string
* akey ``meta``: host-resident opaque metadata
* akey ``kv``: the GPU-resident KV payload

This module intentionally contains no LMCache imports.  ``gds_backend`` owns
the metadata format and GPU allocator; this class only translates a small,
typed Python API into the C shim exported by ``libdaosgdr.so``.
"""

from __future__ import annotations

import ctypes
import functools
import os
from pathlib import Path
from typing import Optional, Union


class DaosObjectError(RuntimeError):
    """A non-zero return code from the DAOS object shim."""

    def __init__(self, operation: str, rc: int):
        super().__init__(f"{operation} failed: rc={rc}")
        self.operation = operation
        self.rc = rc


def _default_library_path() -> str:
    configured = os.environ.get("DAOSGDR_LIB")
    if configured:
        return configured

    # Source-tree execution: lmcache_daos/ is next to libdaosgdr.so.
    sibling = Path(__file__).resolve().parent.parent / "libdaosgdr.so"
    if sibling.is_file():
        return str(sibling)

    # Installed execution: let the dynamic loader search its configured paths.
    return "libdaosgdr.so"


@functools.lru_cache(maxsize=None)
def load_object_library(path: Optional[str] = None) -> ctypes.CDLL:
    library_path = path or _default_library_path()
    lib = ctypes.CDLL(library_path)

    lib.daosgdr_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.daosgdr_init.restype = ctypes.c_void_p

    lib.daosgdr_put.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.daosgdr_put.restype = ctypes.c_int
    if hasattr(lib, "daosgdr_put_device"):
        lib.daosgdr_put_device.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        lib.daosgdr_put_device.restype = ctypes.c_int

    lib.daosgdr_stat.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.daosgdr_stat.restype = ctypes.c_int

    lib.daosgdr_get.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.daosgdr_get.restype = ctypes.c_int
    if hasattr(lib, "daosgdr_get_device"):
        lib.daosgdr_get_device.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        lib.daosgdr_get_device.restype = ctypes.c_int

    lib.daosgdr_remove.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.daosgdr_remove.restype = ctypes.c_int

    lib.daosgdr_fini.argtypes = [ctypes.c_void_p]
    lib.daosgdr_fini.restype = None
    return lib


def _key_bytes(key: Union[str, bytes]) -> bytes:
    encoded = key if isinstance(key, bytes) else key.encode("utf-8")
    if not encoded:
        raise ValueError("DAOS object key must not be empty")
    if b"\x00" in encoded:
        raise ValueError("DAOS object key must not contain NUL")
    return encoded


class DaosObjectStore:
    """Thin owner of one ``libdaosgdr`` context/object handle."""

    def __init__(
        self,
        pool: str,
        container: str,
        library_path: Optional[str] = None,
        *,
        library=None,
        initial_meta_capacity: int = 4096,
        max_meta_capacity: int = 1 << 20,
    ):
        self.library_path = library_path or _default_library_path()
        self._lib = library or load_object_library(self.library_path)
        self._initial_meta_capacity = max(1, int(initial_meta_capacity))
        self._max_meta_capacity = max(
            self._initial_meta_capacity, int(max_meta_capacity)
        )
        self._ctx = self._lib.daosgdr_init(
            pool.encode("utf-8"), container.encode("utf-8")
        )
        if not self._ctx:
            raise RuntimeError(
                f"daosgdr_init(pool={pool!r}, container={container!r}) failed"
            )

    def stat(self, key: Union[str, bytes]) -> Optional[bytes]:
        """Return opaque metadata, or ``None`` when the dkey is absent."""
        encoded = _key_bytes(key)
        capacity = self._initial_meta_capacity
        while True:
            buf = ctypes.create_string_buffer(capacity)
            actual = ctypes.c_size_t(capacity)
            rc = self._lib.daosgdr_stat(
                self._ctx, encoded, buf, ctypes.byref(actual)
            )
            if rc == 0 and actual.value <= capacity:
                if actual.value == 0:
                    return None
                return bytes(buf.raw[: actual.value])

            # The shim reports the required metadata size even for a
            # too-small output buffer. Retry once with that exact capacity.
            if actual.value > capacity and actual.value <= self._max_meta_capacity:
                capacity = actual.value
                continue
            raise DaosObjectError("daosgdr_stat", int(rc))

    def put(
        self,
        key: Union[str, bytes],
        gpu_ptr: int,
        size: int,
        metadata: bytes,
        device_id: int = 0,
    ) -> None:
        if not gpu_ptr or size <= 0:
            raise ValueError("GPU pointer and size must be non-zero")
        if not metadata:
            raise ValueError("object metadata must not be empty")
        meta_buf = ctypes.create_string_buffer(metadata, len(metadata))
        args = (
            self._ctx, _key_bytes(key), ctypes.c_void_p(gpu_ptr),
            ctypes.c_size_t(size), meta_buf, ctypes.c_size_t(len(metadata)),
        )
        if hasattr(self._lib, "daosgdr_put_device"):
            rc = self._lib.daosgdr_put_device(*args, ctypes.c_int(device_id))
        else:
            if device_id != 0:
                raise RuntimeError(
                    "this libdaosgdr.so only supports CUDA device 0; rebuild "
                    "the shim from discos_minji/libdaosgdr.c"
                )
            rc = self._lib.daosgdr_put(*args)
        if rc != 0:
            raise DaosObjectError("daosgdr_put", int(rc))

    def get(
        self, key: Union[str, bytes], gpu_ptr: int, size: int, device_id: int = 0
    ) -> None:
        if not gpu_ptr or size <= 0:
            raise ValueError("GPU pointer and size must be non-zero")
        args = (
            self._ctx, _key_bytes(key), ctypes.c_void_p(gpu_ptr),
            ctypes.c_size_t(size),
        )
        if hasattr(self._lib, "daosgdr_get_device"):
            rc = self._lib.daosgdr_get_device(*args, ctypes.c_int(device_id))
        else:
            if device_id != 0:
                raise RuntimeError(
                    "this libdaosgdr.so only supports CUDA device 0; rebuild "
                    "the shim from discos_minji/libdaosgdr.c"
                )
            rc = self._lib.daosgdr_get(*args)
        if rc != 0:
            raise DaosObjectError("daosgdr_get", int(rc))

    def remove(self, key: Union[str, bytes]) -> bool:
        rc = self._lib.daosgdr_remove(self._ctx, _key_bytes(key))
        if rc != 0:
            raise DaosObjectError("daosgdr_remove", int(rc))
        return True

    def close(self) -> None:
        if self._ctx:
            self._lib.daosgdr_fini(self._ctx)
            self._ctx = None
