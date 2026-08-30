#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
output_path="${1:-${project_dir}/conv_learning.mp4}"

if [[ "${output_path}" != /* ]]; then
    output_path="$(pwd)/${output_path}"
fi

command -v ffmpeg >/dev/null 2>&1 || {
    echo "ffmpeg is required but was not found on PATH." >&2
    exit 1
}

make -C "${project_dir}" all

frames_dir="$(mktemp -d "${TMPDIR:-/tmp}/maze_rl_video.XXXXXX")"
cleanup() {
    rm -rf -- "${frames_dir}"
}
trap cleanup EXIT INT TERM

"${project_dir}/maze_rl" \
    --generalization-video \
    --episodes 5000 \
    --seed 1 \
    --frames "${frames_dir}" \
    --fps 30 \
    --width 1280 \
    --height 720 \
    --seconds 45

manifest="${frames_dir}/manifest.txt"
[[ -f "${manifest}" ]] || {
    echo "Renderer did not write ${manifest}." >&2
    exit 1
}
fps="$(awk -F= '$1 == "fps" { print $2 }' "${manifest}")"
[[ "${fps}" =~ ^[0-9]+$ ]] || {
    echo "Renderer wrote an invalid frame rate." >&2
    exit 1
}

mkdir -p "$(dirname "${output_path}")"
ffmpeg -hide_banner -loglevel warning -y \
    -framerate "${fps}" \
    -i "${frames_dir}/frame_%05d.png" \
    -c:v libx264 \
    -preset medium \
    -crf 18 \
    -pix_fmt yuv420p \
    -movflags +faststart \
    -an \
    "${output_path}"

echo "Wrote ${output_path}"
