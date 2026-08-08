#!/usr/bin/env bash
# Run a command with wall-time and aggregate-RSS watchdogs.
set -euo pipefail

wall_minutes=180
memory_gb=12

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wall-minutes) wall_minutes="$2"; shift 2 ;;
    --memory-gb) memory_gb="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "usage: $0 [--wall-minutes N] [--memory-gb N] -- command [args...]" >&2; exit 2 ;;
  esac
done

[[ $# -gt 0 ]] || { echo "safe_guard: missing command" >&2; exit 2; }

start_seconds="$(date +%s)"
wall_seconds=$((wall_minutes * 60))
memory_kb=$((memory_gb * 1024 * 1024))

"$@" &
child_pid=$!

terminate_tree() {
  pkill -TERM -P "$child_pid" 2>/dev/null || true
  kill -TERM "$child_pid" 2>/dev/null || true
}
trap terminate_tree INT TERM EXIT

while kill -0 "$child_pid" 2>/dev/null; do
  now_seconds="$(date +%s)"
  if (( now_seconds - start_seconds > wall_seconds )); then
    echo "safe_guard: wall-time limit (${wall_minutes} min) exceeded" >&2
    terminate_tree
    wait "$child_pid" 2>/dev/null || true
    exit 124
  fi

  # Read the child and its direct descendants from one process-table snapshot.
  # The monitored command can exit between kill -0 above and this probe.  In
  # that normal race, ps may return nonzero; do not let `set -o pipefail`
  # convert a successfully completed command into a silent watchdog failure.
  rss_kb="$(
    ps -eo pid=,ppid=,rss= 2>/dev/null |
      awk -v child_pid="$child_pid" '
        $1 == child_pid || $2 == child_pid { sum += $3 }
        END { print sum + 0 }
      ' || true
  )"
  rss_kb="${rss_kb:-0}"
  if (( rss_kb > memory_kb )); then
    echo "safe_guard: RSS limit (${memory_gb} GiB) exceeded" >&2
    terminate_tree
    wait "$child_pid" 2>/dev/null || true
    exit 137
  fi
  sleep 5
done

trap - INT TERM EXIT
wait "$child_pid"
