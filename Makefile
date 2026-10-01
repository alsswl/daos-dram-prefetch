DAOS_PREFIX ?= /opt/daos-gds-gpu
CC ?= gcc
CFLAGS ?= -O2

.PHONY: all check

all: libdaosgdr.so

libdaosgdr.so: libdaosgdr.c Makefile
	$(CC) -shared -fPIC $(CFLAGS) -o $@ $< \
		-I$(DAOS_PREFIX)/include \
		-L$(DAOS_PREFIX)/lib64 \
		-ldaos -ldaos_common -lgurt -lcart \
		-Wl,-rpath,$(DAOS_PREFIX)/lib64

check: libdaosgdr.so
	@! ldd ./libdaosgdr.so | grep 'not found'

# Separate experimental artifact; never replace the baseline shared library.
libdaosgdr_scatter.so: libdaosgdr_scatter.c libdaosgdr.c Makefile
	$(CC) -shared -fPIC $(CFLAGS) -Wall -Wextra -o $@ $< \
		-I$(DAOS_PREFIX)/include -L$(DAOS_PREFIX)/lib64 \
		-ldaos -ldaos_common -lgurt -lcart -Wl,-rpath,$(DAOS_PREFIX)/lib64
