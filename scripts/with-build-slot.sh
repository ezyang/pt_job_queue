#!/bin/bash
# Run a command holding one of the workspace's build slots, so only a few
# PyTorch builds run on this host at once, whoever starts them: ptq,
# agents' rebuild.sh, OSS DMZ sandboxes, the nightly seed build. Builds
# share the host with other jobs (and users); too many at once ran it out of
# memory.
#
#   with-build-slot.sh CMD...    wait for a free slot, run CMD, free it when CMD exits
#   with-build-slot.sh --status  who holds the slots (run on the host)
#
# Slots are flock(1) locks on $WORKSPACE/.build-slots/slot-*; there are as
# many slots as files (ptq setup creates them). flock works on read-only
# files, so sandboxes that see the workspace read-only share the same slots.
# The lock is held by this shell, not by CMD, so a daemon a build leaves
# running can't keep a slot.
set -u
WORKSPACE="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
SLOTS_DIR="$WORKSPACE/.build-slots"
DEFAULT_SLOTS=2

if [ ! -e "$SLOTS_DIR/slot-0" ]; then
    mkdir -p "$SLOTS_DIR" 2>/dev/null &&
        for ((i = 0; i < DEFAULT_SLOTS; i++)); do touch "$SLOTS_DIR/slot-$i"; done 2>/dev/null
fi
shopt -s nullglob
slots=("$SLOTS_DIR"/slot-*)

if [ "${1:-}" = "--status" ]; then
    for s in "${slots[@]}"; do
        ino=$(stat -c %i "$s")
        pids=$(awk -v ino=":$ino" '$2 == "FLOCK" && $6 ~ ino"$" {print $5}' /proc/locks | sort -u)
        if [ -z "$pids" ]; then
            echo "$(basename "$s"): free"
        else
            for pid in $pids; do
                echo "$(basename "$s"): $(ps -o pid=,etime=,args= -p "$pid" 2>/dev/null | cut -c1-200) (cwd $(readlink "/proc/$pid/cwd" 2>/dev/null))"
            done
        fi
    done
    exit 0
fi

if [ $# -eq 0 ]; then
    echo "usage: $0 CMD... | --status" >&2
    exit 2
fi
if [ ${#slots[@]} -eq 0 ]; then
    echo "with-build-slot: no build slots in $SLOTS_DIR (run ptq setup on the host); running ungoverned" >&2
    exec "$@"
fi

waited=0
while :; do
    for s in "${slots[@]}"; do
        exec 9<"$s" || continue
        if flock -n 9; then
            [ $waited -gt 0 ] && echo "with-build-slot: got $(basename "$s") after ${waited}s" >&2
            "$@" 9<&-
            exit $?
        fi
        exec 9<&-
    done
    if [ $((waited % 60)) -eq 0 ]; then
        echo "with-build-slot: all ${#slots[@]} build slots are busy; waiting (on the host, $0 --status shows who holds them)" >&2
    fi
    sleep 5
    waited=$((waited + 5))
done
