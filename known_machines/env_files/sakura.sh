#!/bin/bash
module purge

module load gcc/15 openmpi/5.0 hdf5-mpi/2.0.0 cmake/3.30 git/2.50

# bundled deps

# Site-specific install paths (FUKA's HOME_KADATH is read at RUNTIME by libkadath;
# unset -> segfault on FUKA initial-data read). @SITE_FILE@ is filled in by
# `simpilot` setup with the path to your per-user site file; edit that file to
# add or change install dirs.
[ -f "@SITE_FILE@" ] && source "@SITE_FILE@"