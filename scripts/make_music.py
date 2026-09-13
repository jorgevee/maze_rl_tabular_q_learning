#!/usr/bin/env python3
"""Synthesizes the demo video's backing track from scratch (stdlib only).

A drill instrumental: gliding 808 sub, half-time snare, fast hat rolls and a
sparse minor-key pluck motif. Instrumental by construction -- every sound here
is generated from arithmetic, so there are no samples, no third-party audio,
and nothing to attribute. Same premise as the rest of the project.

    python3 scripts/make_music.py out.wav [seconds]
"""

import array
import math
import random
import sys
import wave

RATE = 44100
BPM = 141.0                      # drill sits ~140-145, felt half-time
BEAT = 60.0 / BPM
BAR = BEAT * 4
EIGHTH = BEAT / 2.0
SIXTEENTH = BEAT / 4.0

# G minor. Drill leans on the darker degrees, so the motif favours the b3, b6
# and the leading tone rather than resolving cleanly.
G1, Bb1, C2, D2, Eb2, F2 = 48.999, 58.270, 65.406, 73.416, 77.782, 87.307
G4, A4, Bb4, C5, D5, Eb5, F5, G5 = 392.00, 440.00, 466.16, 523.25, 587.33, 622.25, 698.46, 784.00

# 808 line, one entry per bar of the 4-bar loop:
#   (root, slide_target_or_None, beat_at_which_the_slide_starts)
# The slide is the signature of the genre -- a drill 808 glides between notes
# rather than restriking them.
EIGHT_O_EIGHT = [
    (G1, Bb1, 3.25),
    (F2, None, 0.0),
    (Eb2, D2, 2.75),
    (C2, G1, 3.0),
]

# Sparse 8th-note motif, one row per bar. None is a rest, and the rests matter
# as much as the notes -- drill arrangements are mostly space.
MELODY = [
    [G4, None, Bb4, None, D5, None, C5, None],
    [Bb4, None, None, A4, None, G4, None, None],
    [D5, None, Eb5, None, D5, None, Bb4, None],
    [C5, None, None, Bb4, None, A4, None, None],
]

# Kick placements in 16ths from the bar start; syncopated, locking with the 808.
KICKS = [
    [0, 6, 10],
    [0, 7, 11, 14],
    [0, 6, 10],
    [0, 7, 10, 13],
]

# Hat subdivision per beat: 2 = 8ths, 4 = 16ths, 6 = sextuplet roll,
# 8 = 32nd burst. Rolls are what give drill its forward motion.
HATS = [
    [4, 4, 4, 6],
    [4, 4, 6, 4],
    [4, 4, 4, 8],
    [4, 6, 4, 6],
]

NOISE = None


def noise_table():
    """One deterministic white-noise table, sliced by the percussion voices."""
    global NOISE
    if NOISE is None:
        rng = random.Random(20260911)
        NOISE = [rng.uniform(-1.0, 1.0) for _ in range(RATE)]
    return NOISE


def mix(buf, start, samples, gain):
    n = len(buf)
    for i, value in enumerate(samples):
        j = start + i
        if 0 <= j < n:
            buf[j] += value * gain


def sub808(freq, slide_to, slide_at, duration, drive=2.1):
    """Saturated sine sub with a pitch glide.

    Two pitch moves happen here: a very short drop at the onset, which is what
    gives an 808 its attack, and then an optional smooth glide to a second
    note. Frequency varies over time, so the phase has to be accumulated rather
    than computed from i * w.
    """
    count = int(duration * RATE)
    out = [0.0] * count
    phase = 0.0
    attack, release = int(0.004 * RATE), int(0.05 * RATE)
    norm = math.tanh(drive)
    glide_len = 0.13
    for i in range(count):
        t = i / RATE
        f = freq
        if t < 0.028:                       # onset transient
            f = freq * (1.0 + 3.0 * (1.0 - t / 0.028))
        elif slide_to is not None and t > slide_at:
            u = min(1.0, (t - slide_at) / glide_len)
            u = u * u * (3.0 - 2.0 * u)     # smoothstep, so the glide eases
            f = freq + (slide_to - freq) * u
        phase += 2.0 * math.pi * f / RATE
        env = math.exp(-2.9 * t / duration)
        if i < attack:
            env *= i / attack
        if i > count - release:
            env *= max(0.0, (count - i) / release)
        out[i] = math.tanh(math.sin(phase) * drive) / norm * env
    return out


def kick(duration=0.16):
    """Pitch-swept sine with a click, sitting above the 808's register."""
    count = int(duration * RATE)
    out = [0.0] * count
    phase = 0.0
    for i in range(count):
        t = i / RATE
        f = 46.0 + 145.0 * math.exp(-t / 0.020)
        phase += 2.0 * math.pi * f / RATE
        env = math.exp(-t / 0.055)
        out[i] = math.tanh(math.sin(phase) * 1.9) * env
    return out


def snare(duration=0.26):
    """Noise body plus a tuned tone. Lands on beat 3 for the half-time feel."""
    count = int(duration * RATE)
    table = noise_table()
    out = [0.0] * count
    hp = 0.0
    coeff = math.exp(-2.0 * math.pi * 1400.0 / RATE)
    for i in range(count):
        t = i / RATE
        raw = table[(i * 7 + 311) % len(table)]
        hp = (1.0 - coeff) * raw + coeff * hp
        bright = raw - hp                       # crude highpass
        env = math.exp(-t / 0.075)
        body = math.sin(2.0 * math.pi * 190.0 * t) * math.exp(-t / 0.035)
        out[i] = (bright * 0.85 + body * 0.5) * env
    return out


def hat(duration=0.040, open_hat=False):
    """Short bright noise burst."""
    if open_hat:
        duration = 0.16
    count = int(duration * RATE)
    table = noise_table()
    out = [0.0] * count
    lp = 0.0
    coeff = math.exp(-2.0 * math.pi * 7000.0 / RATE)
    decay = 0.055 if open_hat else 0.011
    for i in range(count):
        raw = table[(i * 13 + 977) % len(table)]
        lp = (1.0 - coeff) * raw + coeff * lp
        out[i] = (raw - lp) * math.exp(-(i / RATE) / decay)
    return out


def pluck(freq, duration=0.42):
    """Dark bell-ish pluck for the motif."""
    count = int(duration * RATE)
    out = [0.0] * count
    w = 2.0 * math.pi * freq / RATE
    attack = int(0.004 * RATE)
    for i in range(count):
        t = i / RATE
        env = math.exp(-t / 0.115)
        if i < attack:
            env *= i / attack
        out[i] = env * (math.sin(w * i)
                        + 0.42 * math.sin(2.0 * w * i) * math.exp(-t / 0.05)
                        + 0.16 * math.sin(3.01 * w * i) * math.exp(-t / 0.03))
    return out


def echo(buf, delay_s, feedback, taps=3):
    """Feedback delay, applied to the melody alone so the low end stays tight."""
    step = int(delay_s * RATE)
    for tap in range(1, taps + 1):
        gain = feedback ** tap
        offset = step * tap
        for i in range(len(buf) - offset):
            buf[i + offset] += buf[i] * gain
    return buf


def render(duration):
    total = int(duration * RATE)
    left, right = [0.0] * total, [0.0] * total
    lead = [0.0] * total
    bars = int(math.ceil(duration / BAR))

    hat_closed, hat_open = hat(), hat(open_hat=True)
    kick_hit, snare_hit = kick(), snare()

    for bar in range(bars):
        slot = bar % 4
        start = int(bar * BAR * RATE)
        # Two-bar intro: motif and hats only, so the 808 lands as a drop.
        dropped = bar >= 2

        for step, freq in enumerate(MELODY[slot]):
            if freq is None:
                continue
            at = start + int(step * EIGHTH * RATE)
            mix(lead, at, pluck(freq), 0.30 if dropped else 0.38)

        if dropped:
            root, slide_to, slide_beat = EIGHT_O_EIGHT[slot]
            sub = sub808(root, slide_to, slide_beat * BEAT, BAR * 0.90)
            mix(left, start, sub, 0.34)
            mix(right, start, sub, 0.34)

            for sixteenth in KICKS[slot]:
                at = start + int(sixteenth * SIXTEENTH * RATE)
                mix(left, at, kick_hit, 0.34)
                mix(right, at, kick_hit, 0.34)

            # Half-time: one snare, on beat 3.
            at = start + int(2.0 * BEAT * RATE)
            mix(left, at, snare_hit, 0.30)
            mix(right, at, snare_hit, 0.30)

        for beat_index, subdivision in enumerate(HATS[slot]):
            for tick in range(subdivision):
                at = start + int((beat_index * BEAT + tick * BEAT / subdivision) * RATE)
                # Accented downbeats, quieter rolls -- a flat hat line sounds
                # mechanical, and the roll is meant to sit under the beat.
                gain = 0.115 if tick == 0 else (0.052 if subdivision > 4 else 0.072)
                voice = hat_open if (subdivision == 4 and tick == 2 and slot == 3) else hat_closed
                pan = 0.5 + (0.10 if tick % 2 else -0.10)
                mix(left, at, voice, gain * (1.0 - pan) * 2.0)
                mix(right, at, voice, gain * pan * 2.0)

    echo(lead, EIGHTH * 0.75, 0.32)
    for i in range(total):
        left[i] += lead[i] * 0.52
        right[i] += lead[i] * 0.48
    return left, right


def finish(channel, fade_out_s):
    """Short fade-in so the beat starts promptly, long fade-out at the end."""
    total = len(channel)
    fade_in = int(0.25 * RATE)
    fade_out = int(fade_out_s * RATE)
    for i in range(total):
        s = channel[i]
        if i < fade_in:
            s *= i / fade_in
        if i > total - fade_out:
            s *= max(0.0, (total - i) / fade_out) ** 1.4
        channel[i] = math.tanh(s * 1.05)
    return channel


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "music.wav"
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 51.0
    fade_out = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0

    left, right = render(duration)
    left = finish(left, fade_out)
    right = finish(right, fade_out)

    frames = array.array("h")
    for l, r in zip(left, right):
        frames.append(int(max(-1.0, min(1.0, l)) * 30000))
        frames.append(int(max(-1.0, min(1.0, r)) * 30000))

    with wave.open(out_path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(frames.tobytes())

    print(f"wrote {out_path}: {duration:.1f}s drill @ {BPM:.0f} BPM, "
          f"{fade_out:.1f}s fade-out")


if __name__ == "__main__":
    main()
