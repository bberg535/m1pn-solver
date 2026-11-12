#!/usr/bin/env bash
# GHS line source with FCS (docs/FCS.md, docs/DLRA.md): full-rank element WENO vs DLRA.
# Usage: run_line_source.sh BIN OUTDIR ELEMENTS DEGREE "ORDERS" "TOLERANCES" [full|nofull] [extra args...]
# Runs full rank (mirror reduction auto = quarter domain, and off = full domain) and DLRA per tolerance.
set -u
BIN=$1; OUT=$2; EL=$3; DEG=$4; ORDERS=$5; TOLS=$6; FULL=${7:-full}; shift 7 || shift $#
EXTRA="$*"
G="--case line-source --dimension 2 --pulse-width .01 --background 9e-4 --sigma-s 3 --time 0.3333333333333333 --stabilization element-weno --fcs analytic --degree $DEG --elements $EL"
mkdir -p "$OUT"
for N in $ORDERS; do
  if [ "$FULL" = full ]; then
    for MR in auto off; do
      d="$OUT/full_m${MR}_e${EL}_p${DEG}_P$N"
      [ -f "$d/summary.json" ] || "$BIN" $G --pn-order $N --mirror-reduction $MR $EXTRA --output "$d" > "$d.log" 2>&1
    done
  fi
  for T in $TOLS; do
    d="$OUT/dlra_t${T}_e${EL}_p${DEG}_P$N"
    [ -f "$d/summary.json" ] || "$BIN" $G --pn-order $N --dlra on --dlra-tolerance $T $EXTRA --output "$d" > "$d.log" 2>&1
  done
done
echo done
