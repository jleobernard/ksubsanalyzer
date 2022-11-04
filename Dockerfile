FROM python:3.10.7-slim AS dependencies-install
WORKDIR /code
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential gcc curl g++ openjdk-8-jdk
COPY ./requirements.txt /code/requirements.txt
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

FROM python:3.10.7-slim
WORKDIR /code/app
COPY --from=dependencies-install /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH "${PYTHONPATH}:/code/app"
COPY ./logging.conf /code/logging.conf
COPY ./app /code/app
CMD ["python3", "main.py"]
