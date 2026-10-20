"""
tts.py - Spoken announcements using macOS's built-in `say` command.

main.py passes the labels detected in each frame to Announcer.announce().
The Announcer tracks when each label was seen and decides what to say:

  * New appearance: when an object comes into view (it was absent for at
    least `forget_after` seconds), it is announced right away. Show a
    glass, lower it, show it again, and you hear "glass" again.
  * Confirmation: a label must appear in `min_frames` detections in a row
    before it is spoken, so a one-frame misdetection stays silent.
  * Cooldown: while an object stays in view, it is repeated at most once
    every `cooldown` seconds instead of every frame.
  * No overlapping speech: `say` runs as a background process so the video
    loop never blocks. While it's talking, nothing new starts. Labels that
    are due stay due and get spoken as soon as it finishes.
"""

import shutil
import subprocess
import time


class Announcer:
    def __init__(self, cooldown, forget_after=1.5, min_frames=1, voice=None, rate=None):
        if shutil.which("say") is None:
            raise RuntimeError("The macOS 'say' command was not found; TTS needs macOS.")

        self.cooldown = cooldown
        self.forget_after = forget_after
        self.min_frames = max(1, int(min_frames))
        self.voice = voice
        self.rate = rate
        self._last_seen = {}    # label -> last time it was detected
        self._appeared_at = {}  # label -> when its current appearance began
        self._last_spoken = {}  # label -> when it was last announced
        self._streak = {}       # label -> detections in a row it appeared in
        self._process = None    # the currently running `say` process, if any

    def is_speaking(self):
        return self._process is not None and self._process.poll() is None

    def announce(self, labels):
        """Update tracking with this frame's labels; speak any that are due.

        Returns the list of labels actually announced (possibly empty).
        """
        now = time.monotonic()
        # dict.fromkeys de-duplicates while keeping detection order,
        # so three people in frame become a single "human".
        labels = list(dict.fromkeys(labels))

        self._streak = {label: self._streak.get(label, 0) + 1 for label in labels}
        for label in labels:
            if now - self._last_seen.get(label, float("-inf")) >= self.forget_after:
                self._appeared_at[label] = now  # (re)entered the view
            self._last_seen[label] = now

        if not labels or self.is_speaking():
            return []

        due = []
        for label in labels:
            if self._streak[label] < self.min_frames:
                continue  # not confirmed yet
            last_spoken = self._last_spoken.get(label, float("-inf"))
            not_yet_announced = last_spoken < self._appeared_at[label]
            if not_yet_announced or now - last_spoken >= self.cooldown:
                due.append(label)
        if not due:
            return []

        for label in due:
            self._last_spoken[label] = now

        self._speak(self._to_sentence(due))
        return due

    def stop(self):
        """Stop any speech that's still playing (called on shutdown)."""
        if self.is_speaking():
            self._process.terminate()
        self._process = None

    @staticmethod
    def _to_sentence(labels):
        if len(labels) == 1:
            return labels[0]
        return ", ".join(labels[:-1]) + " and " + labels[-1]

    def _speak(self, text):
        cmd = ["say"]
        if self.voice:
            cmd += ["-v", self.voice]
        if self.rate:
            cmd += ["-r", str(self.rate)]
        cmd.append(text)
        # Popen (not run) so the call returns immediately.
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
