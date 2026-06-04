#!/bin/bash

module load cmake/4.0 cuda/12.8 gcc/14 openmpi/5.0 openmpi_gpu/5.0 hdf5-mpi/1.12.2 clang/20

# bundled deps

# Site-specific install paths (FUKA's HOME_KADATH is read at RUNTIME by libkadath;
# unset -> segfault on FUKA initial-data read). @SITE_FILE@ is filled in by
# `simpilot` setup with the path to your per-user site file; edit that file to
# add or change install dirs.
[ -f "@SITE_FILE@" ] && source "@SITE_FILE@"