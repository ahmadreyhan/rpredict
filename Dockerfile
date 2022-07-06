# Use the official lightweight Python image
# https://hub.docker.com/_/python
FROM python:3.9.12

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Run the web service on container
#CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
CMD python app.py