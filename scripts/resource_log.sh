#!/usr/bin/env bash
# Samples GPU and host resource use every few seconds into a CSV while you run a call or a load test.
#   bash scripts/resource_log.sh results/resources.csv 2     # file, interval seconds; Ctrl-C to stop
# Columns: time, GPU utilization %, GPU memory used MiB, GPU memory total MiB, GPU power W, host CPU % (all cores),
# host memory used MiB, API processes' resident memory MiB, active calls (from /healthz, this node).
# Use it for the "resource consumption" line of the proof of concept: run it, place calls, stop it, then
#   python scripts/resource_log.py summarize results/resources.csv   (or open the CSV)
set -euo pipefail
OUT="${1:-resources.csv}"; EVERY="${2:-2}"
echo "time,gpu_util_pct,gpu_mem_used_mib,gpu_mem_total_mib,gpu_power_w,cpu_pct,host_mem_used_mib,api_rss_mib,active_calls" > "$OUT"
prev_idle=0; prev_total=0
while true; do
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo ",,,")
  read -r _ u n s i io irq sirq _ < /proc/stat
  idle=$((i + io)); total=$((u + n + s + i + io + irq + sirq))
  cpu=$(( prev_total > 0 ? 100 * (total - prev_total - (idle - prev_idle)) / (total - prev_total) : 0 ))
  prev_idle=$idle; prev_total=$total
  mem=$(free -m | awk '/Mem:/ {print $3}')
  rss=$(ps -eo rss,args | awk '/uvicorn api.main/ && !/awk/ {s+=$1} END {print int(s/1024)}')
  act=$(curl -fsS --max-time 1 http://127.0.0.1:8080/healthz 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin).get("active_calls",""))' 2>/dev/null || echo "")
  echo "$(date +%T),${gpu:-,,,},$cpu,$mem,${rss:-0},$act" >> "$OUT"
  sleep "$EVERY"
done
