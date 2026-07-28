"""ONNX model loading and session management."""

import contextlib
import threading
import time
from collections.abc import Generator
from pathlib import Path

import onnxruntime as ort

from .log import logger


def _create_session(model_path: Path) -> ort.InferenceSession:
    """Load an ONNX model into an InferenceSession."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4

    available_providers = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    logger.info(
        "Loading ONNX model path={} size_mb={:.2f} available_providers={} selected_providers={}",
        model_path,
        model_path.stat().st_size / 1e6,
        available_providers,
        providers,
    )
    session = ort.InferenceSession(model_path, sess_options, providers=providers)
    logger.info(
        "ONNX session ready path={} inputs={} outputs={}",
        model_path,
        [i.name for i in session.get_inputs()],
        [o.name for o in session.get_outputs()],
    )
    return session


class ModelSession:
    IDLE_TIMEOUT = 60.0 * 3

    def __init__(self, path: Path, name: str) -> None:
        self._path = path
        self._name = name
        self._file_state: tuple[float, bool] | None = None  # (mtime, healthy)
        self._session: ort.InferenceSession | None = None
        self._session_lock = threading.Lock()

        self._ref_count = 0
        self._ref_count_lock = threading.Lock()
        self._last_use_ended = 0.0
        self._idle_event = threading.Event()
        self._thread = threading.Thread(target=self._idle_session_monitor, daemon=True)
        self._thread.start()

    def _load(self) -> ort.InferenceSession | None:
        with self._session_lock:
            if self._session is not None:
                return self._session

            try:
                self._session = _create_session(self._path)
            except Exception:
                logger.exception("Failed to load ONNX session name={} path={}", self._name, self._path)
                return None
            return self._session

    def is_available(self) -> bool:
        if not self._path.exists():
            return False
        if self._session is not None:
            return True

        mtime = self._path.stat().st_mtime
        if self._file_state is None or self._file_state[0] != mtime:
            healthy = self._load() is not None
            self._file_state = (mtime, healthy)
        return self._file_state[1]

    @contextlib.contextmanager
    def load(self) -> Generator[ort.InferenceSession]:
        if not self.is_available():
            raise RuntimeError(f"{self._name} model not loaded")

        session = self._load()
        if session is None:
            raise RuntimeError(f"Failed to load {self._name} model")

        with self._ref_count_lock:
            self._ref_count += 1
        try:
            yield session
        finally:
            self._last_use_ended = time.monotonic()
            with self._ref_count_lock:
                self._ref_count -= 1
                if self._ref_count == 0:
                    self._idle_event.set()

    def _idle_session_monitor(self) -> None:
        while True:
            self._idle_event.wait()
            self._idle_event.clear()

            deadline = self._last_use_ended + self.IDLE_TIMEOUT
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

            with self._ref_count_lock:
                if self._ref_count > 0:
                    continue
            if time.monotonic() < self._last_use_ended + self.IDLE_TIMEOUT:
                continue

            with self._session_lock:
                if self._session is not None:
                    logger.info("Unloading idle ONNX session name={} path={}", self._name, self._path)
                    self._session = None


class ModelLoader:
    """Manages ONNX model session for single-stage 3-class inference."""

    def __init__(self, model_path: Path) -> None:
        self._model_session = ModelSession(model_path, "CNN+FFT 3-class")

    @property
    def model_available(self) -> bool:
        """Check if single 3-class model is available."""
        return self._model_session is not None and self._model_session.is_available()

    def get_session(
        self,
    ) -> contextlib.AbstractContextManager[ort.InferenceSession]:
        """Get single 3-class model session."""
        return self._model_session.load()
