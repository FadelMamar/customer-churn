ARG PYTHON_VERSION=3.12
FROM python:$PYTHON_VERSION-slim


WORKDIR /app

# Copy only requirements first for caching optimization
COPY ./app/requirements.txt .

# Install litserve and requirements
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app .

RUN mkdir /models
VOLUME /models

EXPOSE 4141
CMD ["python", "/app/app.py"]
