ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
COPY kern reference.json workload.toml bench16k test16k /opt/kern-eval/
COPY nccl/ /opt/kern-eval/nccl/
RUN chmod +x /opt/kern-eval/kern /opt/kern-eval/bench16k /opt/kern-eval/test16k && ln -s /opt/kern-eval/bench16k /usr/local/bin/bench16k && ln -s /opt/kern-eval/test16k /usr/local/bin/test16k && mkdir -p /home/worker && chmod 1777 /home/worker
RUN python3 -m venv /opt/humanize && /opt/humanize/bin/pip install --no-cache-dir 'git+https://github.com/humanfia/humanize2.git@413d02e44d0cc0514b9f5bd3fcefea156b047a49'
ENV PATH=/opt/humanize/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/kern-eval/nccl:/usr/local/cuda/lib64
WORKDIR /workspace
ENTRYPOINT []
CMD ["sleep", "infinity"]
