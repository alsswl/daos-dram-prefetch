#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
gcc -shared -fPIC -O2 -Wall -Wextra -Werror \
  -isystem /opt/daos-gds-gpu/include "$experiment_dir/lmcache_transport.c" \
  -L/opt/daos-gds-gpu/lib64 -ldaos -lgurt -o "$experiment_dir/lmcache_transport.so"
