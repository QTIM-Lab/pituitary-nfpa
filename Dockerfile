FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

WORKDIR /

ENV TZ=America/Montreal
ARG DEBIAN_FRONTEND=noninteractive

# Add a /.local/bin and /.local/lib directories to allow editable python
# installs by any user, and /.cache, which is used by pre-commit
RUN mkdir -p -m 777 /.local/bin /.local/lib /.cache

# Install python and other useful programs
RUN apt update && apt install -y \
        python3 \
        python3-pip \
        python3-venv \
        python-is-python3 \
        git \
        htop \
        vim \
        tmux \
        curl && \
    apt clean

# Add the bashrc to start up the container correctly for local development
COPY docker/container_bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

# Copy in the slurm entrypoint file
COPY slurm/slurm_entrypoint.sh /

# Install Python requirements using the compiled version of the requirements
COPY qms_data/compiled_requirements_dev.txt ./
RUN pip install --no-cache-dir --upgrade pip==24.3.1 && \
    pip install --no-cache-dir setuptools==75.6.0  && \
    pip install --no-cache-dir -r compiled_requirements_dev.txt && \
    rm compiled_requirements_dev.txt

# Add a /.local/bin and /.local/lib directories to allow editable python
# installs by any user, and /.cache, which is used by pre-commit
RUN mkdir -p -m 777 /.local/bin /.local/lib /.cache

# Add the bashrc to start up the container correctly in development mode
COPY docker/container_bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

# Build the current commit hash of the repo into the container
ARG BUILD_COMMIT_HASH=nohash
RUN echo "${BUILD_COMMIT_HASH}" > /etc/docker-build-git-hash
