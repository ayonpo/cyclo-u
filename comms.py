"""Utility functions for platform TTS/STT.

This module is primarily for Android/Termux use. When Termux isn't
available the functions fall back to safer desktop-friendly behavior.
"""

import subprocess
import os


def listen():
    """Attempt to retrieve speech-to-text output via Termux, fallback safe.

    Returns a decoded string from stdout/stderr, or empty string on failure.
    """
    try:
        stt = subprocess.Popen(
            ["termux-speech-to-text"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        # Termux probably not available on this platform; return empty.
        return ""

    try:
        out, err = stt.communicate(timeout=5)
    except Exception:
        # If communication times out or fails, try to terminate process
        try:
            stt.terminate()
        except Exception:
            pass
        out, err = "", ""

    # Prefer stdout; ensure we always return str
    if out:
        return out if isinstance(out, str) else str(out)
    if err:
        return err if isinstance(err, str) else str(err)
    return ""


def speak(words):
    try:
        os.system("termux-tts-speak '{}'".format(str(words).replace("'", "\\'")))
    except Exception:
        # Try subprocess call; if Termux not present this will raise and we
        # fallback to printing which is safe on desktops.
        command = ["termux-tts-speak", str(words)]
        try:
            subprocess.run(command, check=True)
        except Exception:
            print(words)


def listen2():
    try:
        inp = subprocess.getoutput("termux-speech-to-text")
        return str(inp)
    except Exception:
        return ""
