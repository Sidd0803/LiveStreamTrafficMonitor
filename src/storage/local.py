"""Local JSONL backend -- the Phase 1 store, and the one the tests run against.

**Why JSONL and not a single JSON array.** The producer is a poller that writes
continuously and may be killed at any moment. Rewriting a whole JSON document
on every observation means every write is a window in which the entire history
can be lost to a partial serialization; an append is a window in which at worst
the newest line is truncated. The failure mode of the format we chose costs one
observation. The failure mode of the one we did not costs the dataset.

It is also the format that survives being read by something else -- `tail`,
`wc -l`, pandas, a shell one-liner -- while the poller is still writing, which
matters a lot more during a hackathon than schema elegance does.

**Why full scans are fine here.** Every read walks the file. That is a
deliberate choice, not an oversight: a 250-camera sweep every few minutes for a
full day is on the order of 10^5 lines, which parses in well under a second,
and Phase 3 replaces this with indexed Firestore queries anyway. Building an
index over an append-only file for a demo-scale dataset would be effort spent
on the implementation we are about to throw away. If this ever gets slow, the
fix is to finish Phase 3, not to optimize this file.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from src import config
from src.storage.base import Observation, Store

log = logging.getLogger(__name__)

# Not in config.py: this is an implementation detail of one backend, not a
# tunable anyone is expected to change, and config.py is reserved for
# thresholds that Phase 2 tunes against the eval set.
DEFAULT_FILENAME = "observations.jsonl"


class LocalStore:
    """Append-only JSONL store under `config.RUNS_DIR`.

    Satisfies the `Store` protocol. `path` is injectable so tests (and one-off
    scripts) can write to a temp directory instead of polluting `runs/`.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            resolved = Path(config.RUNS_DIR) / DEFAULT_FILENAME
        else:
            resolved = Path(path)
            # Being handed a directory is the obvious mistake (pytest's tmp_path
            # is one). Treat it as "put the default file in here" rather than
            # failing later with an IsADirectoryError from open().
            if resolved.is_dir():
                resolved = resolved / DEFAULT_FILENAME
        self.path: Path = resolved

        # A sweep runs SWEEP_CONCURRENCY workers, so save() can be called from
        # several threads. Appends are not reliably atomic on Windows, and two
        # interleaved partial writes would produce two corrupt lines instead of
        # zero. The lock is per-instance, which covers the real case (one store
        # shared by the pipeline); two LocalStore objects on the same path from
        # different threads would still race, and are not a supported usage.
        self._lock = threading.Lock()

    # --- writing -----------------------------------------------------------

    def save(self, obs: Observation) -> None:
        self.save_many([obs])

    def save_many(self, observations: list[Observation]) -> None:
        """Append a batch in a single open/flush.

        Batching matters for a sweep: hundreds of individual open-append-close
        cycles is a lot of syscalls for no benefit, and one flush at the end
        narrows the window in which a crash can truncate a line.
        """
        if not observations:
            return

        # Created on demand rather than at import time, so importing this module
        # never has a side effect on the filesystem.
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize before opening the file. A record that fails to serialize
        # then raises without having written a partial line.
        lines = [json.dumps(obs.to_dict(), ensure_ascii=False) for obs in observations]

        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                for line in lines:
                    fh.write(line + "\n")
                fh.flush()

        log.debug("appended %d observation(s) to %s", len(lines), self.path)

    # --- reading -----------------------------------------------------------

    def _iter_all(self) -> list[Observation]:
        """Every parseable observation in the file, in file (append) order.

        A missing file is normal, not an error: the dashboard is expected to
        start before the poller has written anything, and it should render an
        empty map rather than a traceback.

        Bad lines are skipped with a warning. The realistic cause is a poller
        killed mid-write leaving a truncated final line, but a line mangled any
        other way is handled the same. Losing one observation is acceptable;
        losing the whole dashboard to one bad byte is not -- and an exception
        raised here would be raised on every subsequent refresh, permanently.
        """
        if not self.path.exists():
            log.debug("no observation file at %s yet", self.path)
            return []

        observations: list[Observation] = []
        skipped = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    observations.append(Observation.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    # Truncated JSON, a valid-JSON line of the wrong shape, and
                    # an unparseable timestamp all land here.
                    skipped += 1
                    log.warning(
                        "skipping unreadable observation at %s:%d (%s)",
                        self.path.name, lineno, exc,
                    )

        if skipped:
            log.warning(
                "%d of %d lines in %s were unreadable",
                skipped, skipped + len(observations), self.path.name,
            )
        return observations

    def recent(
        self, camera_id: str | None = None, limit: int = 500
    ) -> list[Observation]:
        """Newest first, optionally for one camera, capped at `limit`.

        Filtering before sorting keeps the per-camera detail view cheap; sorting
        by `captured_at` rather than trusting append order is what makes this
        correct for a concurrent sweep, where workers finish out of order.
        """
        rows = self._iter_all()
        if camera_id is not None:
            rows = [o for o in rows if o.camera_id == camera_id]

        # Stable sort, so observations sharing a timestamp keep their file order
        # instead of shuffling between refreshes.
        rows.sort(key=lambda o: o.captured_at, reverse=True)
        return rows[: max(0, limit)]

    def latest_per_camera(self) -> list[Observation]:
        """The newest observation for each camera -- one row per camera.

        This runs on every dashboard refresh to draw the map, so it is a single
        pass with a dict keyed by camera_id: O(n) over the file, O(cameras) in
        memory, no sort of the full history. See the module docstring for why a
        full scan (rather than an index or a tail-read) is the right call at
        this scale.

        Compares timestamps rather than taking the last line per camera, since
        a concurrent sweep does not append in chronological order.
        """
        newest: dict[str, Observation] = {}
        for obs in self._iter_all():
            current = newest.get(obs.camera_id)
            if current is None or obs.captured_at > current.captured_at:
                newest[obs.camera_id] = obs

        # Sorted for a stable render order; the map redraws every refresh and
        # arbitrary reordering is visible to the user.
        return sorted(newest.values(), key=lambda o: o.camera_id)


def get_store(backend: str | None = None) -> Store:
    """Return the configured store. This is the Phase 3 seam.

    Everything downstream depends on the `Store` protocol and calls this, so
    moving to GCP is `STORAGE_BACKEND=gcp` in the environment plus one new
    module -- no changes to the pipeline or the dashboard.

    `backend` is an override for tests and scripts; it defaults to
    `config.STORAGE_BACKEND`.
    """
    name = (backend if backend is not None else config.STORAGE_BACKEND).strip().lower()

    if name == "local":
        return LocalStore()

    if name == "gcp":
        # Phase 3. Explicit rather than a silent fallback to local: a
        # misconfigured deployment quietly writing observations to a container
        # filesystem that vanishes on the next Cloud Run scale-to-zero is a
        # much worse outcome than refusing to start.
        raise NotImplementedError(
            "STORAGE_BACKEND='gcp' is Phase 3 and not built yet "
            "(src/storage/gcp.py does not exist). Use STORAGE_BACKEND='local'."
        )

    raise ValueError(
        f"unknown STORAGE_BACKEND {name!r}; expected 'local' or 'gcp'"
    )
