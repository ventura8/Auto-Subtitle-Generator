"""NVIDIA Canary/Parakeet ASR engine support.

Nothing here imports ``modules.models`` or ``modules.pipeline``: ``modules/__init__.py``
imports the models eagerly, so either import would be a cycle. Cues leave this package as
plain ``(start, end, text)`` tuples and the transcription stage maps them to segments.
"""
