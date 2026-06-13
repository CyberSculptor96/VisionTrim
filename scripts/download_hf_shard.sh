#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
    echo "Usage: $0 <url> <output-file> <total-bytes> [chunk-bytes] [jobs]" >&2
    exit 1
fi

url=$1
out=$2
total=$3
chunk=${4:-33554432}
jobs=${5:-8}
proxy=${CURL_PROXY:-http://127.0.0.1:7890}
curl_bin=${CURL_BIN:-}

if [[ -z "$curl_bin" ]]; then
    if command -v curl >/dev/null 2>&1; then
        curl_bin=$(command -v curl)
    elif [[ -x /mnt/pfs/pynr16/Shichao_Bao/miniconda3/bin/curl ]]; then
        curl_bin=/mnt/pfs/pynr16/Shichao_Bao/miniconda3/bin/curl
    else
        echo "curl not found; set CURL_BIN=/path/to/curl" >&2
        exit 127
    fi
fi

mkdir -p "$(dirname "$out")"

if [[ -f "$out" ]]; then
    size=$(stat -c%s "$out")
    if [[ "$size" == "$total" ]]; then
        echo "Already complete: $out"
        exit 0
    fi
fi

parts_dir="${out}.parts"
manifest="$parts_dir/manifest.tsv"
mkdir -p "$parts_dir"
: > "$manifest"

idx=0
start=0
while (( start < total )); do
    end=$(( start + chunk - 1 ))
    if (( end >= total )); then
        end=$(( total - 1 ))
    fi
    expected=$(( end - start + 1 ))
    printf '%06d\t%d\t%d\t%d\n' "$idx" "$start" "$end" "$expected" >> "$manifest"
    start=$(( end + 1 ))
    idx=$(( idx + 1 ))
done

download_one() {
    local idx=$1
    local start=$2
    local end=$3
    local expected=$4
    local part="$parts_dir/part-$idx"
    local tmp="$part.tmp.$$"
    local size

    if [[ -f "$part" ]]; then
        size=$(stat -c%s "$part" 2>/dev/null || echo 0)
        if [[ "$size" == "$expected" ]]; then
            return 0
        fi
    fi

    rm -f "$tmp"
    "$curl_bin" --proxy "$proxy" -L --fail --retry 10 --retry-all-errors --retry-delay 2 \
        --connect-timeout 30 --speed-time 120 --speed-limit 1024 \
        --range "${start}-${end}" -sS -o "$tmp" "$url"

    size=$(stat -c%s "$tmp")
    if [[ "$size" != "$expected" ]]; then
        echo "Bad chunk $idx: expected $expected bytes, got $size" >&2
        exit 2
    fi
    mv "$tmp" "$part"
}

export -f download_one
export url parts_dir proxy curl_bin

awk -F '\t' '{print $1, $2, $3, $4}' "$manifest" |
    xargs -P "$jobs" -n 4 bash -c 'download_one "$@"' _

while IFS=$'\t' read -r idx _ _ expected; do
    part="$parts_dir/part-$idx"
    if [[ ! -f "$part" ]]; then
        echo "Missing chunk $idx" >&2
        exit 3
    fi
    size=$(stat -c%s "$part")
    if [[ "$size" != "$expected" ]]; then
        echo "Bad saved chunk $idx: expected $expected bytes, got $size" >&2
        exit 4
    fi
done < "$manifest"

tmp_out="$out.tmp"
: > "$tmp_out"
while IFS=$'\t' read -r idx _ _ _; do
    cat "$parts_dir/part-$idx" >> "$tmp_out"
done < "$manifest"

size=$(stat -c%s "$tmp_out")
if [[ "$size" != "$total" ]]; then
    echo "Bad assembled file: expected $total bytes, got $size" >&2
    exit 5
fi

mv "$tmp_out" "$out"
echo "Complete: $out"
