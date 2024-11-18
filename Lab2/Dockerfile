# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

# Install pip requirements
COPY requirements.txt .
RUN python -m venv venv
# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg
RUN python -m pip install -r requirements.txt

RUN /bin/bash -c "source venv/bin/activate"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

