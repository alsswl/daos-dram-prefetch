#!/usr/bin/env python3
"""Process-local NUMA interleave, inherited by exec/children; no host changes."""
import ctypes
import json
import os
import sys


def interleave():
    lib = ctypes.CDLL('libnuma.so.1', use_errno=True)
    lib.set_mempolicy.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong]
    lib.set_mempolicy.restype = ctypes.c_int
    mask = ctypes.c_ulong(3)  # Nodes 0 and 1 on client-5.
    if lib.set_mempolicy(3, ctypes.byref(mask), 64) != 0:  # MPOL_INTERLEAVE
        raise OSError(ctypes.get_errno(), 'set_mempolicy interleave(0,1) failed')
    result = current_policy()
    assert result['mode'] == 3 and result['node_mask'] == 3
    return result


def current_policy():
    lib = ctypes.CDLL('libnuma.so.1', use_errno=True)
    lib.get_mempolicy.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ulong),
                                  ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong]
    lib.get_mempolicy.restype = ctypes.c_int
    policy, actual = ctypes.c_int(), ctypes.c_ulong()
    if lib.get_mempolicy(ctypes.byref(policy), ctypes.byref(actual), 64, None, 0) != 0:
        raise OSError(ctypes.get_errno(), 'get_mempolicy failed')
    return dict(mode=policy.value, node_mask=actual.value, pid=os.getpid())


if __name__ == '__main__':
    print(json.dumps(interleave()), flush=True)
    args = sys.argv[1:]
    if args and args[0] == '--':
        args.pop(0)
    if not args:
        raise SystemExit('command required')
    os.execvp(args[0], args)
