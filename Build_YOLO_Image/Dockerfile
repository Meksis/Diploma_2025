FROM ultralytics/ultralytics:latest
# FROM ultralytics/ultralytics@sha256:22730a04e8b54f867c37c76fe9a2a93a1368204d2d92a0e10c3285128d3338ed

ENV POETRY_VERSION=2.1.2 
RUN curl -sSL https://install.python-poetry.org | python3 - --version ${POETRY_VERSION}
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /diploma_code

COPY ./Weather_Distortions_Recognition/pyproject.toml ./Weather_Distortions_Recognition/poetry.lock ./app_code/
WORKDIR /diploma_code/app_code

RUN poetry config virtualenvs.in-project true
RUN poetry install --no-interaction --no-root

COPY ./Weather_Distortions_Recognition ./ 
CMD ["bash"]