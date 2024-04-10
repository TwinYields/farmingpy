FROM mcr.microsoft.com/devcontainers/base:jammy

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    libcurl4-openssl-dev \
    curl \
    python3-pip \
    python3-dev \
    python3-setuptools\
    r-base \
    r-base-dev \
    build-essential \
    cmake swig \
    libgit2-dev libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
    software-properties-common && \
    rm -rf /var/lib/apt/lists/*

#RUN add-apt-repository ppa:ubuntugis/ppa
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev


# DOTNET and APSIMX
# https://github.com/dotnet/dotnet-docker/blob/main/documentation/scenarios/installing-dotnet.md
RUN apt-get update && apt-get install -y --no-install-recommends \
        # .NET dependencies
        libc6 \
        libgcc-s1 \
        libgssapi-krb5-2 \
        libicu70 \
        libssl3 \
        libstdc++6 \
        zlib1g \
        libunwind8 \
        liblttng-ust1 \
        libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://dot.net/v1/dotnet-install.sh | bash /dev/stdin -Channel 8.0 -InstallDir /usr/share/dotnet \
    && ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet

WORKDIR /opt/app
RUN git clone --depth 1 -b clock_management https://github.com/mpastell/ApsimX
RUN dotnet build -o /usr/local/lib/apsimx -c Release ApsimX/Models/Models.csproj

# R packages for rpy2 and EUPTF2 interface
RUN R -e "options(warn=2); install.packages(c('pak', 'glue', 'Rdpack'), dependencies=TRUE)"
RUN R -e "options(warn=2); install.packages(c('data.table', 'plyr', 'ranger'), dependencies=TRUE)"
RUN R -e "options(warn=2); pak::pkg_install('tkdweber/euptf2')"

COPY docker/requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install -U pip setuptools
RUN pip install -r requirements.txt
COPY requirements.txt /opt/app/fp/fp_requirements.txt
RUN pip install -r /opt/app/fp/fp_requirements.txt

USER vscode
RUN echo "export PYTHONPATH=/usr/local/lib/apsimx" >> ~/.bashrc

WORKDIR /workspaces/
USER vscode