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
# The lock is held by a flock process that waits for CMD (-o: CMD doesn't
# inherit it, so a daemon a build leaves running can't keep a slot), and
# /proc/locks names that process, which --status shows.
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
                # The flock process. Its child is the sh wrapper below, whose
                # argv is: sh -c SCRIPT sh WAITED SLOT CMD...
                child=$(pgrep -P "$pid" | head -1)
                argv=()
                [ -n "$child" ] && mapfile -d '' -t argv < "/proc/$child/cmdline"
                cwd=$(readlink "/proc/$child/cwd" 2>/dev/null)
                echo "$(basename "$s"): pid $pid, for $(ps -o etime= -p "$pid" | tr -d ' '): ${argv[*]:6}${cwd:+ (in $cwd)}" | cut -c1-300
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

BUSY=213  # flock's exit status when the slot is taken; CMD's own 213 is mapped to 1
waited=0
while :; do
    for s in "${slots[@]}"; do
        flock -n -o -E $BUSY "$s" sh -c '
            [ "$1" -gt 0 ] && echo "with-build-slot: got $2 after $1s" >&2
            shift 2; "$@"; rc=$?; [ $rc -eq '$BUSY' ] && rc=1; exit $rc' sh "$waited" "$(basename "$s")" "$@"
        rc=$?
        [ $rc -ne $BUSY ] && exit $rc
    done
    if [ $((waited % 60)) -eq 0 ]; then
        echo "with-build-slot: all ${#slots[@]} build slots are busy; waiting (on the host, $0 --status shows who holds them)" >&2
    fi
    sleep 5
    waited=$((waited + 5))
done
