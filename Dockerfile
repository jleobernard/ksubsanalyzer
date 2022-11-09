FROM python:3.10.7-slim
WORKDIR /code/app
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y --no-install-recommends
#RUN apt-get install -y --no-install-recommends build-essential gcc curl g++ openjdk-8-jdk
RUN apt-get install -y --no-install-recommends default-jre
COPY ./requirements.txt /code/requirements.txt
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
ENV PYTHONPATH "${PYTHONPATH}:/code/app"
ENV JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
COPY ./logging.conf /code/logging.conf
COPY ./app /code/app
CMD ["python3", "main.py"]
