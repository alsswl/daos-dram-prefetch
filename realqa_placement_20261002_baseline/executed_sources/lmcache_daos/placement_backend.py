"""Opt-in CXS placement experiment; existing backends keep their addressing.

A frozen manifest owns the OID and placement keys. Position metadata is recorded
when ChunkedTokenDatabase yields full keys, without changing key identity.
Stored-key positions are persisted locally for key-only removal after restart.
Both control and treatment use the same position tracking and transport code.
"""
import ctypes as C
import functools
import json
from pathlib import Path
import sqlite3
import threading

from .capacity_pipeline_backend import CapacityPipelineBackend
from .gds_backend import _cfg
from .object_binding import DaosObjectError
from experiments.chunk_placement.addressing import Addressing

_position_cache={}
_cache_lock=threading.RLock()


def clear_positions():
    with _cache_lock: _position_cache.clear()


def install_positions():
    from lmcache.v1.token_database import ChunkedTokenDatabase
    from lmcache.utils import CacheEngineKey
    if getattr(ChunkedTokenDatabase, '_placement_positions', False): return
    original=ChunkedTokenDatabase.process_tokens
    @functools.wraps(original)
    def process(db,*args,**kwargs):
        for start,end,key in original(db,*args,**kwargs):
            if isinstance(key,CacheEngineKey):
                if start % db.chunk_size: raise ValueError('Placement needs aligned absolute chunk positions')
                identity=key.to_string();index=start//db.chunk_size
                with _cache_lock:
                    previous=_position_cache.setdefault(identity,index)
                    if previous!=index:raise ValueError('Conflicting positions for same full KV key')
            yield start,end,key
    ChunkedTokenDatabase.process_tokens=process
    ChunkedTokenDatabase._placement_positions=True


def load_library(path):
    lib=C.CDLL(str(path))
    specs={
        'placement_open':[C.c_char_p,C.c_char_p,C.c_uint64,C.POINTER(C.c_void_p),C.POINTER(C.c_uint64),C.POINTER(C.c_uint64)],
        'placement_shard':[C.c_void_p,C.c_char_p,C.POINTER(C.c_uint)],
        'placement_io':[C.c_void_p,C.c_char_p,C.c_char_p,C.c_void_p,C.c_size_t,C.c_int],
        'placement_put_meta':[C.c_void_p,C.c_char_p,C.c_char_p,C.c_char_p,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_int],
        'placement_meta':[C.c_void_p,C.c_char_p,C.c_char_p,C.c_void_p,C.POINTER(C.c_size_t)],
        'placement_remove':[C.c_void_p,C.c_char_p,C.c_char_p,C.c_char_p],
        'placement_close':[C.c_void_p,C.c_int],
    }
    for name,types in specs.items():
        getattr(lib,name).argtypes=types;getattr(lib,name).restype=C.c_int
    return lib


def checked(rc):
    if rc: raise DaosObjectError('placement transport',rc)


class PlacementObjectStore:
    def __init__(self,manifest_path,pool,container,library_path=None):
        self.manifest_path=Path(manifest_path)
        self.manifest=json.loads(self.manifest_path.read_text())
        m=self.manifest
        if m['pool']!=pool or m['container']!=container or m['version']!=1:
            raise ValueError('Placement manifest mismatch')
        self.scheme=Addressing.from_dict(m['addressing'])
        self._lib=load_library(m['library']); self._ctx=C.c_void_p()
        hi,lo=C.c_uint64(),C.c_uint64()
        checked(self._lib.placement_open(pool.encode(),container.encode(),m['nonce'],C.byref(self._ctx),C.byref(hi),C.byref(lo)))
        try:
            if f'{hi.value}.{lo.value}'!=m['oid']: raise ValueError('Object layout/OID changed')
            for i,key in enumerate(self.scheme.placement_keys):
                shard=C.c_uint();checked(self._lib.placement_shard(self._ctx,key.encode(),C.byref(shard)))
                if shard.value!=i: raise ValueError('Placement key no longer maps to expected shard')
        except BaseException:
            self.close();raise

    def address(self,locator):
        full_key,sep,index=locator.rpartition('\x1f')
        if not sep: raise ValueError('Missing absolute chunk position')
        dk,ak=self.scheme.address(full_key,int(index))
        mk=b'meta' if self.scheme.mode=='baseline' else b'meta:'+full_key.encode()
        return dk,ak,mk

    def put(self,key,gpu_ptr,size,metadata,device_id=0):
        dk,ak,mk=self.address(key);buf=C.create_string_buffer(metadata,len(metadata))
        checked(self._lib.placement_put_meta(self._ctx,dk,ak,mk,gpu_ptr,size,buf,len(metadata),device_id))

    def stat(self,key):
        dk,_,mk=self.address(key);capacity=4096
        while True:
            buf=C.create_string_buffer(capacity);size=C.c_size_t(capacity)
            rc=self._lib.placement_meta(self._ctx,dk,mk,buf,C.byref(size))
            if size.value>capacity and size.value<=2**20:
                capacity=size.value;continue
            checked(rc)
            if size.value>capacity: raise ValueError('Oversized metadata')
            return bytes(buf.raw[:size.value]) if size.value else None

    def get(self,key,gpu_ptr,size,device_id=0):
        if device_id!=0: raise ValueError('Placement experiment is single-GPU')
        dk,ak,_=self.address(key)
        checked(self._lib.placement_io(self._ctx,dk,ak,gpu_ptr,size,0))

    def remove(self,key):
        dk,ak,mk=self.address(key)
        checked(self._lib.placement_remove(self._ctx,dk,ak,mk));return True

    def close(self):
        if self._ctx:
            ctx=self._ctx;self._ctx=None
            checked(self._lib.placement_close(ctx,0))


class PlacementBackend(CapacityPipelineBackend):
    def __init__(self,config,*args,**kwargs):
        manifest=_cfg(config,'placement_manifest')
        if not manifest: raise ValueError('placement_manifest required')
        if config.use_layerwise or config.enable_async_loading:
            raise ValueError('Placement experiment requires synchronous non-layerwise path')
        self.placement_manifest=str(Path(manifest).resolve())
        self._position_lock=threading.RLock()
        self._positions=sqlite3.connect(str(Path(manifest).with_suffix('.positions.sqlite')),timeout=30,check_same_thread=False)
        self._positions.execute('PRAGMA journal_mode=WAL')
        self._positions.execute('CREATE TABLE IF NOT EXISTS positions (key TEXT PRIMARY KEY, idx INTEGER NOT NULL)')
        self._positions.commit()
        install_positions()
        try: super().__init__(config,*args,**kwargs)
        except BaseException:
            self._positions.close();raise
        self._staging_trace.emit('placement_enabled',mode=self._object.scheme.mode,
                                 oid=self._object.manifest['oid'],shards=len(self._object.scheme.placement_keys))

    def _create_object_store(self,**kwargs):
        return PlacementObjectStore(self.placement_manifest,**kwargs)

    def _object_key(self,key):
        full=self.object_namespace+key.to_string()
        with _cache_lock:index=_position_cache.get(key.to_string())
        if index is None:
            with self._position_lock:
                row=self._positions.execute('SELECT idx FROM positions WHERE key=?',(full,)).fetchone()
            if row is None: raise ValueError('Key-only operation lacks persisted chunk position')
            index=row[0]
        return full+'\x1f'+str(index)

    def batched_submit_put_task(self,keys,memory_objs,*args,**kwargs):
        positions=[]
        for key in keys:
            full,_,index=self._object_key(key).rpartition('\x1f');positions.append((full,int(index)))
        with self._position_lock, self._positions:
            for full,index in positions:
                previous=self._positions.execute('SELECT idx FROM positions WHERE key=?',(full,)).fetchone()
                if previous is not None and previous[0]!=index:
                    raise ValueError('Same logical KV key has conflicting absolute positions')
            self._positions.executemany('INSERT OR IGNORE INTO positions VALUES (?,?)',positions)
        return super().batched_submit_put_task(keys,memory_objs,*args,**kwargs)

    def close(self):
        try: super().close()
        finally: self._positions.close()


def configure(cfg,manifest):
    cfg['extra_config'].update({
        'storage_plugin.daosgds.module_path':'lmcache_daos.placement_backend',
        'storage_plugin.daosgds.class_name':'PlacementBackend',
        'daosgds.placement_manifest':str(Path(manifest).resolve())})
