"""ctypes binding for the separate array/SGL shim (no framework imports)."""
import ctypes as C

from .object_binding import DaosObjectStore, DaosObjectError, _key_bytes
from .scatter_plan import MAX_IOVS


class ScatterObjectStore(DaosObjectStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            get = self._lib.daosgdr_getv_device
            put = self._lib.daosgdr_putv_device
            common = [C.c_void_p, C.c_char_p, C.POINTER(C.c_void_p),
                      C.POINTER(C.c_size_t), C.POINTER(C.c_uint64), C.c_uint]
            # Python fake libraries used in tests need not expose C signatures.
            if isinstance(get, C._CFuncPtr):
                get.argtypes = common + [C.c_int]
                get.restype = C.c_int
                put.argtypes = common + [C.c_void_p, C.c_size_t, C.c_int]
                put.restype = C.c_int
        except Exception:
            self.close()
            raise

    def _vectors(self, segments, device_id):
        if not self._ctx:
            raise RuntimeError("Object store is closed")
        if type(device_id) is not int or device_id < 0:
            raise ValueError("Invalid CUDA device")
        if not 0 < len(segments) <= MAX_IOVS:
            raise ValueError("Invalid SGL size")
        end = 0
        for s in segments:
            if (any(type(v) is not int for v in (s.pointer, s.length, s.offset))
                    or s.pointer <= 0 or s.length <= 0 or s.offset < end
                    or s.offset + s.length >= 2**64
                    or s.pointer + s.length >= 2**64):
                raise ValueError("Invalid or overlapping segment")
            end = s.offset + s.length
        n = len(segments)
        return ((C.c_void_p * n)(*(s.pointer for s in segments)),
                (C.c_size_t * n)(*(s.length for s in segments)),
                (C.c_uint64 * n)(*(s.offset for s in segments)), C.c_uint(n))

    def getv(self, key, segments, device_id=0):
        vectors = self._vectors(segments, device_id)
        rc = self._lib.daosgdr_getv_device(
            self._ctx, _key_bytes(key), *vectors, C.c_int(device_id))
        if rc:
            raise DaosObjectError("daosgdr_getv", int(rc))

    def putv(self, key, segments, metadata, device_id=0):
        vectors = self._vectors(segments, device_id)
        end = 0
        for s in segments:
            if s.offset != end:
                raise ValueError("Store must cover the complete chunk")
            end += s.length
        if not metadata:
            raise ValueError("Metadata is required")
        meta = C.create_string_buffer(metadata, len(metadata))
        rc = self._lib.daosgdr_putv_device(
            self._ctx, _key_bytes(key), *vectors, meta,
            C.c_size_t(len(metadata)), C.c_int(device_id))
        if rc:
            raise DaosObjectError("daosgdr_putv", int(rc))
