FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash ca-certificates curl git libgomp1 tmux vim wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check \
    && python -c "import numpy as np, torch; torch.from_numpy(np.zeros(1)).numpy()"

CMD ["bash"]
