#!/usr/bin/env bash
# gpu_run.sh - single-tenant wrapper for GPU-heavy runs on a box shared with the @hermes session.
#
#   scripts/gpu_run.sh "<what>" [timeout_s] -- cmd args...
#
# Protocol (agreed with @hermes, implemented once in /opt/data/gpu_lock.sh):
#   lock = directory /opt/data/gpu.lock holding an `owner` file: "<agent> <pid> <what> <epoch-until>"
#   stale = holder pid dead, or until < now, or dir older than 15 min; every transition is appended to
#   /opt/data/gpu.audit.log.  This wrapper only *uses* that helper - it never deletes the lock itself,
#   so a genuinely stuck lock is always cleared by the two of us together, not blind.
set -u

WHAT="${1:-gpu work}"
TMO="${2:-300}"
shift 2 2>/dev/null || shift $#
[ "${1:-}" = "--" ] && shift

ME="${FERRO_AGENT_NAME:-autoresearcher}"
HELPER=/opt/data/gpu_lock.sh

if [ ! -f "$HELPER" ]; then
    echo "[gpu_run] $HELPER missing: refusing to start a GPU run without the shared lock" >&2
    exit 78
fi
# shellcheck disable=SC1090
. "$HELPER"

if ! gpu_acquire "$ME" "$WHAT" 15; then
    echo "[gpu_run] busy: another agent holds the GPU; not starting '$WHAT'" >&2
    exit 75
fi
trap 'gpu_release "$ME"' EXIT INT TERM

# hard guard: a stray compute process (not the long-lived serve) means contention - refuse, do not run.
# An orphan surviving a killed wrapper is exactly what silently halved two of my earlier measurements.
stray=""
for pid in $(pgrep -x ferro-embed 2>/dev/null); do
    case "$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)" in
        *" serve "*) : ;;                       # the serve is expected and idle
        *) stray="$stray $pid" ;;
    esac
done
if [ -n "$stray" ]; then
    echo "[gpu_run] REFUSING: another ferro-embed compute process is running:$stray" >&2
    echo "[gpu_run] fix: kill it (kill -9$stray) and re-run; never measure with two engines live" >&2
    gpu_release "$ME"
    exit 76
fi

start=$(date +%s)
timeout "$TMO" "$@"
rc=$?
echo "[gpu_run] '$WHAT' rc=$rc in $(( $(date +%s) - start ))s" >&2
exit $rc
