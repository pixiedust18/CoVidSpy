# aithermometer docker image 
FROM nvidia/cuda:10.0-cudnn7-devel

# get deps
RUN apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
python3-dev python3-pip git g++ wget make libprotobuf-dev protobuf-compiler libopencv-dev \
libgoogle-glog-dev libboost-all-dev libcaffe-cuda-dev libhdf5-dev libatlas-base-dev build-essential ffmpeg libjsoncpp-dev vim net-tools iputils-ping psmisc ssh \
libpqxx-dev libpq-dev libssl-dev autoconf automake git-core libass-dev python3-setuptools \
libfreetype6-dev libsdl2-dev \
pkg-config texinfo zlib1g-dev \ 
libtool libva-dev libvdpau-dev libvorbis-dev nasm libx264-dev \
libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev \
pkg-config texinfo exiftool 

# for python api
RUN pip3 install numpy opencv-python ffmpeg-python psycopg2 Pillow flirimageextractor loguru

# replace cmake as old version has CUDA variable bugs
RUN wget https://github.com/Kitware/CMake/releases/download/v3.16.0/cmake-3.16.0-Linux-x86_64.tar.gz && \
tar xzf cmake-3.16.0-Linux-x86_64.tar.gz -C /opt && \
rm cmake-3.16.0-Linux-x86_64.tar.gz
ENV PATH="/opt/cmake-3.16.0-Linux-x86_64/bin:${PATH}"

# get openpose
WORKDIR /fever_detect
WORKDIR /fever_detect/openpose
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git .

# build it
WORKDIR /fever_detect/openpose/build
RUN cmake -DBUILD_PYTHON=ON .. && make -j `nproc`
RUN make install
WORKDIR /fever_detect
WORKDIR /fever_detect/samples
WORKDIR /fever_detect/results

WORKDIR /fever_detect

# copy files
COPY src/fever_detect.py /fever_detect
COPY src/test.sh /fever_detect
COPY samples/* /fever_detect/samples/





