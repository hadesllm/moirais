#!/bin/bash
# morie Console — double-click to open Terminal ready to run morie.
clear
cat <<'BANNER'
  morie is ready. Some things to try:

    morie --help
    morie list-modules
    morie tutorial

BANNER
exec /bin/bash -l
