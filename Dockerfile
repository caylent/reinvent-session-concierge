FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /usr/src/app

COPY requirements.txt .
RUN apt-get update \
	&& apt-get -y install libpq-dev gcc \
	&& pip install --no-cache-dir -r requirements.txt


COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
