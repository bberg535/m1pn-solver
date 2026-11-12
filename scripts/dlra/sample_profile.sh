#!/usr/bin/env bash
# Poor man's profiler: sample the main thread stack of a running command N times.
# Usage: sample_profile.sh SAMPLES OUTFILE -- command...
N=$1; OUT=$2; shift 3
"$@" > /dev/null 2>&1 &
PID=$!
sleep 3
: > "$OUT"
for i in $(seq "$N"); do
  kill -0 $PID 2>/dev/null || break
  eu-stack -p $PID -1 2>/dev/null | grep -E "^#" | head -14 | sed 's/^#[0-9]* *0x[0-9a-f]* //' | tr '\n' ';' >> "$OUT"
  echo >> "$OUT"
  sleep 0.2
done
kill $PID 2>/dev/null
wait $PID 2>/dev/null
echo "samples: $(wc -l < "$OUT")"
