"""
hypnos/gate/cognition_gate.py — Awake / Dream switch

A threading-based idle monitor that transitions between awake and dream
modes based on input activity.  Uses callbacks to trigger dream start,
step, and end events without coupling to the dream implementation.
"""

import time
import threading
from enum import Enum
from typing import Callable, Optional


class CognitionMode(Enum):
    AWAKE = "awake"
    DREAM = "dream"
    TRANSITIONING = "transitioning"


class CognitionGate:
    """
    Cognition gate: monitors idle time, triggers dream mode.

    Runs a background thread that checks every 1 second whether
    the idle threshold has been exceeded.  All mode transitions
    are thread-safe via a Lock.
    """

    def __init__(
        self,
        idle_threshold: float = 30.0,
        on_dream_start: Optional[Callable] = None,
        on_dream_end: Optional[Callable] = None,
        on_dream_step: Optional[Callable] = None,
        verbose: bool = True,
    ):
        self.idle_threshold = idle_threshold
        self.on_dream_start = on_dream_start
        self.on_dream_end = on_dream_end
        self.on_dream_step = on_dream_step
        self.verbose = verbose

        self._last_input_time = time.time()
        self._mode = CognitionMode.AWAKE
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._dream_thread: Optional[threading.Thread] = None

    # ── public API ────────────────────────────────────────────────────

    def notify_input(self):
        """Signal that user input was received — wake up if dreaming."""
        with self._lock:
            was_dreaming = self._mode == CognitionMode.DREAM
            self._last_input_time = time.time()
            self._mode = CognitionMode.AWAKE

        if was_dreaming:
            if self.verbose:
                print("[Gate] Input received — waking from dream.")
            if self.on_dream_end:
                self.on_dream_end()

    def start(self):
        """Start the background idle monitor thread."""
        self._stop_event.clear()
        self._dream_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="hypnos-gate",
        )
        self._dream_thread.start()
        if self.verbose:
            print(
                f"[Gate] Started. Dream threshold: {self.idle_threshold}s"
            )

    def stop(self):
        """Stop the background monitor."""
        self._stop_event.set()
        if self._dream_thread:
            self._dream_thread.join(timeout=5.0)
        if self.verbose:
            print("[Gate] Stopped.")

    # ── state queries ─────────────────────────────────────────────────

    def is_awake(self) -> bool:
        with self._lock:
            return self._mode == CognitionMode.AWAKE

    def is_dreaming(self) -> bool:
        with self._lock:
            return self._mode == CognitionMode.DREAM

    @property
    def mode(self) -> CognitionMode:
        with self._lock:
            return self._mode

    @property
    def idle_seconds(self) -> float:
        """Seconds since last input."""
        return time.time() - self._last_input_time

    # ── background monitor ────────────────────────────────────────────

    def _monitor_loop(self):
        """Background loop: checks idle time every second."""
        while not self._stop_event.is_set():
            time.sleep(1.0)

            with self._lock:
                idle_time = time.time() - self._last_input_time
                current_mode = self._mode

            if (
                current_mode == CognitionMode.AWAKE
                and idle_time >= self.idle_threshold
            ):
                self._enter_dream_mode()

            elif current_mode == CognitionMode.DREAM:
                if self.on_dream_step:
                    try:
                        self.on_dream_step()
                    except Exception as e:
                        print(f"[Gate] Dream step error: {e}")

    def _enter_dream_mode(self):
        """Transition from awake to dream mode."""
        with self._lock:
            self._mode = CognitionMode.DREAM

        if self.verbose:
            print("[Gate] Idle threshold reached — entering dream mode.")
        if self.on_dream_start:
            self.on_dream_start()
