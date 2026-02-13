FROM distil-whisper-tarsila-base:0.6

ENV PYTHONUNBUFFERED=1 \
    NCCL_P2P_DISABLE="1" \
    NCCL_IB_DISABLE="1" \
    NCCL_DEBUG="INFO"

WORKDIR /app

COPY ./distil-whisper-train.py ./distil-whisper-train.py
