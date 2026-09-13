#!/usr/bin/env bash
# Renders the PPO generalization demo video: trains from scratch, writes
# numbered PNG frames, then assembles them with ffmpeg. Deterministic and
# repeatable -- rerun after any code change to regenerate the video.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
output_path="${1:-${project_dir}/assets/ppo_generalization.mp4}"
width="${WIDTH:-1920}"
height="${HEIGHT:-1080}"
seed="${SEED:-1}"
# Backing track level. 0 disables audio entirely.
music_gain="${MUSIC_GAIN:-0.62}"

if [[ "${output_path}" != /* ]]; then
    output_path="$(pwd)/${output_path}"
fi

command -v ffmpeg >/dev/null 2>&1 || {
    echo "ffmpeg is required but was not found on PATH." >&2
    exit 1
}

make -C "${project_dir}" all

frames_dir="$(mktemp -d "${TMPDIR:-/tmp}/ppo_demo.XXXXXX")"
cleanup() { rm -rf -- "${frames_dir}"; }
trap cleanup EXIT INT TERM

echo "Training and rendering frames (this trains from scratch, a few minutes)..."
"${project_dir}/maze_rl" \
    --render3d --train \
    --seed "${seed}" \
    --min-separation 10 \
    --width "${width}" \
    --height "${height}" \
    --frames "${frames_dir}"

manifest="${frames_dir}/manifest.txt"
[[ -f "${manifest}" ]] || { echo "Renderer did not write ${manifest}." >&2; exit 1; }
fps="$(awk -F= '$1 == "fps" { print $2 }' "${manifest}")"
[[ "${fps}" =~ ^[0-9]+$ ]] || { echo "Invalid frame rate in manifest." >&2; exit 1; }

mkdir -p "$(dirname "${output_path}")"
# yuv420p + faststart so it plays inline on GitHub and social platforms.
silent="${frames_dir}/silent.mp4"
ffmpeg -hide_banner -loglevel warning -y \
    -framerate "${fps}" \
    -i "${frames_dir}/frame_%05d.png" \
    -c:v libx264 -preset slow -crf 20 \
    -pix_fmt yuv420p -movflags +faststart -an \
    "${silent}"

if [[ "${music_gain}" == "0" ]]; then
    mv "${silent}" "${output_path}"
else
    # The backing track is synthesized here rather than sourced, so the video
    # carries no third-party audio and no attribution requirement.
    duration="$(ffprobe -v error -show_entries format=duration \
        -of default=nw=1:nk=1 "${silent}")"
    track="${frames_dir}/track.wav"
    python3 "${script_dir}/make_music.py" "${track}" "${duration}" "${MUSIC_FADE:-6.0}"
    ffmpeg -hide_banner -loglevel warning -y -i "${silent}" -i "${track}" \
        -filter_complex "[1:a]highpass=f=25,volume=${music_gain}[a]" \
        -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k -shortest \
        -movflags +faststart "${output_path}"
fi

echo "Wrote ${output_path}"
ls -lh "${output_path}" | awk '{print "  size:", $5}'
