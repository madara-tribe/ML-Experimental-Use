FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV USER_NAME=developer

# ── Base tools ────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo \
        gnupg2 \
        lsb-release \
        build-essential \
        software-properties-common \
        cmake \
        git \
        curl \
        wget \
        tmux \
        vim \
        udev \
        v4l-utils \
        locales \
        bash-completion \
        tzdata && \
    locale-gen en_US.UTF-8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# ── Python ────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python3-numpy \
        python3-setuptools \
        python3-pytest && \
    pip3 install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ── OpenCV ────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopencv-dev \
        python3-opencv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ── User ──────────────────────────────────────────────────────────
RUN useradd -m -d /home/${USER_NAME} ${USER_NAME} \
        -p $(perl -e 'print crypt("${USER_NAME}", "robot"),"\n"') && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    usermod -aG video ${USER_NAME}

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}
ENV HOME=/home/${USER_NAME}

# ── ROS2 setup script ─────────────────────────────────────────────
COPY scripts/ros2_setup_scripts_ubuntu.sh /home/${USER_NAME}/ros2_setup_scripts_ubuntu.sh

RUN sed -e 's/^\(CHOOSE_ROS_DISTRO=.*\)/#\1\nCHOOSE_ROS_DISTRO=$ROS_DISTRO/g' \
        -i /home/${USER_NAME}/ros2_setup_scripts_ubuntu.sh && \
    bash /home/${USER_NAME}/ros2_setup_scripts_ubuntu.sh && \
    sudo rm -rf /var/lib/apt/lists/*

# ── ROS2 camera dependencies ──────────────────────────────────────
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-cv-bridge \
        ros-${ROS_DISTRO}-image-transport \
        ros-${ROS_DISTRO}-sensor-msgs && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

# ── udev rules (reference only inside container) ──────────────────
COPY setup_udev/99-brio100.rules       /home/${USER_NAME}/udev/99-brio100.rules
COPY setup_udev/brio100-setup@.service /home/${USER_NAME}/udev/brio100-setup@.service

# ── Entrypoint ────────────────────────────────────────────────────
COPY scripts/ros_entrypoint.sh /ros_entrypoint.sh
RUN sudo chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]

WORKDIR /ros2_ws
CMD ["/bin/bash"]
