FROM anibali/pytorch:cuda-10.1
USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \ 
    wget \
    zsh \
    git \
    vim \
    rake \
    tmux \
    gawk \
    perl \
    sed \
    gcc \
    openslide-tools \
    htop \
    g++ \
    parallel


# Install .oh-my-zsh command-line helper (https://github.com/ohmyzsh/ohmyzsh)
RUN wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O - | zsh || true
RUN echo "alias data=\"cd /export/home/data\"" >> ~/.zshrc
RUN echo "alias code=\"cd /export/home/code\"" >> ~/.zshrc
RUN echo "alias med=\"cd /export/medical_ai\"" >> ~/.zshrc
RUN echo "export PYTHONPATH=\"${PYTHONPATH}:/export/home/code/py-wsi\"" >> ~/.zshrc

# Install Janus Vim (https://github.com/carlhuda/janus)
RUN curl -L https://bit.ly/janus-bootstrap -0 | bash
#RUN curl -L https://bit.ly/janus-bootstrap -O | bash
RUN echo "set bg=dark" >> ~/.vimrc

# Install oh-my-tmux tmux configuration gpakosz/.tmux (https://github.com/edd/dev-docker/blob/master/Dockerfile)
WORKDIR /home/user
RUN git clone https://github.com/gpakosz/.tmux.git
RUN ln -s -f .tmux/.tmux.conf
RUN cp .tmux/.tmux.conf.local .

# Conda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN rm -rf /var/lib/apt/lists/*

WORKDIR /root
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version
RUN conda install jupyter
RUN conda install ipython
RUN conda install pip

# Precision Oncology
RUN pip install pillow six scikit-image tensorboardX matplotlib pandas argparse scipy opencv-python openslide-python shapely lmdb tensorboard h5py

# Py-wsi
RUN pip install scikit-image joblib opencv-python
RUN rm -rf /usr/lib/python3/dist-packages/PIL/
RUN pip install --upgrade --force-reinstall pillow

# Bulky
RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
RUN echo "alias make_pytorch=\"conda install pytorch torchvision cudatoolkit=10.1 -c pytorch\"" >>~/.zshrc

# Others
RUN pip install sklearn boxsdk xlrd
RUN echo "alias gto=\"cd /export/home/code/metamind/precision_oncology\"" >>~/.zshrc

# Py-wsi & OpenCV
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
