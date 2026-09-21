ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
COPY kern reference.json reference.parquet workload.toml bench16k test16k judge16k withgpu mix16k /opt/kern-eval/
COPY nccl/ /opt/kern-eval/nccl/
RUN chmod +x /opt/kern-eval/kern /opt/kern-eval/bench16k /opt/kern-eval/test16k /opt/kern-eval/judge16k && ln -s /opt/kern-eval/bench16k /usr/local/bin/bench16k && ln -s /opt/kern-eval/test16k /usr/local/bin/test16k && ln -s /opt/kern-eval/judge16k /usr/local/bin/judge16k && chmod +x /opt/kern-eval/withgpu /opt/kern-eval/mix16k && ln -s /opt/kern-eval/withgpu /usr/local/bin/withgpu && ln -s /opt/kern-eval/mix16k /usr/local/bin/mix16k && mkdir -p /home/worker && chmod 1777 /home/worker
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-pip git ca-certificates curl && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/humanize && /opt/humanize/bin/pip install --no-cache-dir 'git+https://github.com/humanfia/humanize2.git@413d02e44d0cc0514b9f5bd3fcefea156b047a49' 'deepseek-harness-sdk>=0.1.1rc1,<0.1.2'
ENV PATH=/opt/humanize/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/kern-eval/nccl:/usr/local/cuda/lib64
WORKDIR /workspace
ENTRYPOINT []
CMD ["sleep", "infinity"]
