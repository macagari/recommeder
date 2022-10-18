FROM python:3.9-slim
ARG GATEWAY
ENV GATEWAY=$GATEWAY
ENV PYTHONUNBUFFERED=0
EXPOSE 8080
RUN apt update && apt install -y unzip zip curl wget build-essential
RUN pip install --no-cache-dir --upgrade gdown && \
    gdown --fuzzy https://drive.google.com/file/d/1Ao3fHWomRbWC3VoXtRcyk4jhYo1KQ1VY/view -O faire_data.zip && \
    unzip faire_data.zip -d /faire_data && \
    rm faire_data.zip
RUN mkdir -p /root/.cache/torch/hub/checkpoints
RUN cd /root/.cache/torch/hub/checkpoints
#RUN wget https://download.pytorch.org/models/alexnet-owt-7be5be79.pth
#RUN wget https://download.pytorch.org/models/googlenet-1378be20.pth
#RUN wget https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth
#RUN wget https://download.pytorch.org/models/resnet18-f37072fd.pth
#RUN  wget https://download.pytorch.org/models/vgg16-397923af.pth
RUN  wget https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth
ADD ./requirements.txt /
RUN pip install -r /requirements.txt
ADD . /plugin
ENV PYTHONPATH=$PYTHONPATH:/plugin
WORKDIR /plugin/services
CMD echo '$(pwd)'
CMD ["uvicorn", "services_loko:app", "--host", "0.0.0.0", "--port", "8080"]