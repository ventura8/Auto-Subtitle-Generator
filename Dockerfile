# =============================================================================
# Auto Subtitle Generator — Reproducible GPU Docker Image
# =============================================================================
#
# Build:
#   docker image build --tag auto-sub-gen .
#
# Run (GPU):
#   docker run --gpus all --mount='type=volume,source=auto-sub-gen,target=/app/models' --mount='type=bind,source=/path/to/videos,target=/app/input' auto-sub-gen /app/input/video.mkv
#
# =============================================================================

FROM nvcr.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04@sha256:bcf8f5037535884fffbde1c1584af29e9eccc3f432d1cb05a5216a1184af12d8

WORKDIR /app

# cache OS dependencies
# To find the <x> version: docker container run --rm nvcr.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 bash -c "apt-get update -qq && apt-cache policy <x>"
RUN <<EOF
	set -e
	apt-get update
	apt-get install -y --no-install-recommends \
		gcc=4:13.2.0-7ubuntu1 \
		python3=3.12.3-0ubuntu2.1 \
		python3-dev=3.12.3-0ubuntu2.1 \
		python3-pip=24.0+dfsg-1ubuntu1.3 \
		ffmpeg=7:6.1.1-3ubuntu5
	rm -rf /var/lib/apt/lists/*
EOF

# stop python from erroring when installing packages system wide
RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED

# cache pip dependencies (strictly pinned for a reproducible GPU build;
# the repo no longer ships a requirements.txt, so the pins live here)
RUN <<EOF
	set -e
	# faster-whisper depends on plain "onnxruntime", while audio-separator[gpu] depends on "onnxruntime-gpu".
	# Both share the same Python namespace — whichever installs last overwrites the other's native library.
	# If plain "onnxruntime" wins, CUDAExecutionProvider is silently unavailable.
	# To prevent this we install faster-whisper separately with --no-deps and add its unique transitive
	# dependencies (ctranslate2, av) explicitly.
	#
	# onnxruntime-gpu must be pinned to <=1.26.0 (CUDA 12); 1.27.0 switched to nvidia-*-cu13
	# and its native libs won't load on the CUDA 12.8 base image, breaking both VAD
	# (faster-whisper) and vocal separation (audio-separator).
	pip install --no-cache-dir \
		--timeout 300 \
		--retries 10 \
		--extra-index-url https://download.pytorch.org/whl/cu128 \
		torch==2.7.0 \
		torchvision==0.22.0 \
		torchaudio==2.7.0 \
		transformers==4.51.3 \
		sentencepiece==0.2.0 \
		accelerate==1.6.0 \
		onnxruntime-gpu==1.26.0 \
		"audio-separator[gpu]==0.30.2" \
		pyyaml==6.0.2 \
		ctranslate2==4.7.1 \
		av==17.0.1
	pip install --no-cache-dir --timeout 300 --retries 10 --no-deps faster-whisper==1.1.1
EOF
RUN pip install --no-cache-dir pytest pytest-cov

# application
COPY auto_subtitle.py .
COPY config.yaml .
COPY modules/ modules/
# transformers v4.x uses torch_dtype=, not dtype=
RUN sed -i 's/dtype=dtype,/torch_dtype=dtype,/g' modules/models.py

# test
COPY pytest.ini .
COPY tests/ tests/
# test_load_nvidia_paths_torch_fail exercises a Windows-only DLL-path helper
# (os.add_dll_directory / Lib/site-packages) and, on Linux, loads the real
# onnxruntime-gpu which raises ValueError (not ImportError) and escapes the
# helper's `except ImportError` guard. It's irrelevant on Linux, so skip it.
RUN python3 -m pytest -k "not test_load_nvidia_paths_torch_fail"

# Disable upstream's Windows/CUDA-13 cuBLAS DLL probe for this Linux/CUDA-12
# image: _prepare_whisper_cuda13_runtime() probes cublas64_13.dll and raises
# if absent. On Linux the cu12 cuBLAS is resolved by the dynamic linker, so the
# probe must not run. Applied after the test gate so the Windows-oriented tests
# still exercise upstream's source unchanged.
RUN sed -i -E 's|^([[:space:]]*)_prepare_whisper_cuda13_runtime\(\)|\1pass  # Linux: cu12 cuBLAS via dynamic linker; cu13 DLL prep is Windows-only|' modules/models.py

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HF_HOME=/app/models/huggingface

ENTRYPOINT ["python3", "auto_subtitle.py"]
