FROM ubuntu:20.04

WORKDIR /usr/src/app

COPY requirements.txt ./

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install wget to fetch Miniconda
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda on x86 or ARM platforms
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

RUN conda --version

RUN conda create -n python38 python=3.8

RUN echo "conda activate python38" >> ~/.bashrc

RUN /bin/bash -c "source ~/.bashrc"

RUN conda config --env --set always_yes true

RUN pip install -r requirements.txt

COPY . .

VOLUME /usr/src/app/checkpoints

ENTRYPOINT ["python", "main.py"]

# Command to run with volume
# docker run -v ./checkpoints:/usr/src/app/checkpoints app --train;