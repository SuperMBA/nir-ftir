FROM mambaorg/micromamba:1.5.7

# 1) Поставим git (и сертификаты), это нужно для postCreateCommand и git внутри контейнера
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates openssh-client && \
    rm -rf /var/lib/apt/lists/*

# 2) Рабочая папка и conda-окружение по environment.yml
WORKDIR /work/nir-ftir
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba create -y -n ftir311 -f /tmp/environment.yml && \
    micromamba clean -a -y

# 3) Активация conda в шагах RUN
SHELL ["/bin/bash", "-lc"]
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV CONDA_DEFAULT_ENV=ftir311
ENV PATH=/opt/conda/envs/ftir311/bin:$PATH

# 4) Ядро Jupyter (без падения, если расширений нет)
USER $MAMBA_USER
RUN python -m ipykernel install --user --name ftir311 --display-name "Python (ftir311)" || true && \
    jupyter nbextension enable --py widgetsnbextension || true
