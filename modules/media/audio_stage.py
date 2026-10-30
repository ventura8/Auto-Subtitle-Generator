"""Private copies of work-directory audio for the model readers.

The per-video work directory sits inside the input folder, so whoever owns that
folder can list it and keep writing into it while a model loads (minutes for
BS-Roformer, Whisper or Canary). Anything a model or FFmpeg later opened there
by pathname could by then be a link to another file: a planted link at the
predictable Audio-Separator stem name would be written through, and a swapped
``*_temp.wav`` would feed someone else's recording into the subtitles.

An ``AudioStage`` is a private directory under the user's temp location
(``tempfile.mkdtemp``: ``0700``, outside the input tree), created per
transcription. Work-directory audio enters it once, *before* any model load,
by copying the bytes from a descriptor opened without following links
(``input_binding.open_bound_input``); Audio-Separator writes its stems there,
and chunk inputs, chunk stems and the joined vocal track are produced there.
The work directory only receives finished files (``safe_io.copy_file_into``)
so a later run can resume; the readers never go back to it.
"""

import contextlib
import os
import shutil
import tempfile

from modules.media.input_binding import open_bound_input
from modules.safe_io import COPY_CHUNK_BYTES

STAGE_PREFIX = ".asg-stage-"


class AudioStage:
    """A private directory holding the audio the models read during one transcription."""

    def __init__(self, root):
        self.root = root
        self._imported: dict[str, str] = {}

    def path(self, name):
        """Return the stage path for the file name of ``name``."""
        return os.path.join(self.root, os.path.basename(name))

    def new_dir(self, tag):
        """Create a private sub-directory (for a tool that picks its own output names)."""
        return tempfile.mkdtemp(prefix=f"{tag}-", dir=self.root)

    def import_file(self, work_path):
        """Copy ``work_path`` into the stage through a no-follow descriptor and return the copy.

        A link, junction or non-regular file at ``work_path`` (or at a directory
        above it, below the selected input folder) raises ``InputRefusedError``.
        A file is imported once per stage.
        """
        key = os.path.abspath(work_path)
        if key not in self._imported:
            self._imported[key] = self._copy_bound(key)
        return self._imported[key]

    def _copy_bound(self, work_path):
        """Copy the bytes of the descriptor-bound ``work_path`` into a new stage file."""
        target = self.path(work_path)
        bound = open_bound_input(work_path)
        try:
            source = os.fdopen(bound.fd, "rb")
            bound.fd = None
        finally:
            bound.close()
        with source:
            copy = open(target, "xb")
            try:
                with copy:
                    shutil.copyfileobj(source, copy, COPY_CHUNK_BYTES)
            except BaseException:
                # Only the file this call created: an existing entry made "xb" fail above.
                _remove_quietly(target)
                raise
        return target


def _remove_quietly(path):
    """Remove a partial stage copy so a retry can import the same file again."""
    with contextlib.suppress(OSError):
        os.remove(path)


@contextlib.contextmanager
def open_stage():
    """Yield a fresh ``AudioStage``; the directory and everything in it go when the block ends.

    The tree removed is the one ``mkdtemp`` created for this process under the
    user's temp location, never anything inside the input folder.
    """
    root = tempfile.mkdtemp(prefix=STAGE_PREFIX)
    try:
        yield AudioStage(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
