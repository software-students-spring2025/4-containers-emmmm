# in Docker, it is common to base a new image on a previously-created image

FROM python:3.10-slim
# Set the working directory within the image to /app
WORKDIR /app
# the ADD command is how you add files from your local machine into a Docker image

# Copy the current directory contents into the container at /app
ADD . .
# Install any needed packages specified in requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 8000

# Run app.py when the container launches
CMD ["python3", "app.py"]
