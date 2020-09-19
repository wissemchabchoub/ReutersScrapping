FROM tensorflow/tensorflow:2.1.0-gpu-py3-jupyter

SHELL ["/bin/bash", "-c"]

COPY settings/ /opt/settings/


RUN pip install -r /opt/settings/requirements.txt
RUN echo "root:root" | chpasswd

RUN jupyter notebook --generate-config && \
    cp -r /opt/settings/jupyter_notebook_config.py ~/.jupyter/

ARG PYTHON_VERSION="3.6"
ARG RAPIDS_VERSION="0.12"

# setting proxy
ENV http_proxy http://rd-srv-proxy.priv.atos.fr:3128
ENV https_proxy http://rd-srv-proxy.priv.atos.fr:3128

RUN pip install -r /opt/settings/requirements_2.txt

RUN pip install torch==1.5.0 torchvision==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN python3 -m nltk.downloader punkt

WORKDIR /opt/notebook
CMD ["/bin/bash", "-c", "source ~/.profile; source ~/.bashrc; jupyter notebook --allow-root"]






