#!/usr/bin/env python3
"""Use the same 512GiB validation with an opt-in bounded-registration allocator."""
import os
os.environ['DAOS_SEGMENTED_PINNED'] = '1'
from segmented_pinned import install
install()
from pinned_512_probe import main
main()
