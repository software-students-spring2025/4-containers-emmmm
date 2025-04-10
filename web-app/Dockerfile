FROM python:3.13-alpine

# Set the working directory in the container.
WORKDIR /usr/src/app

# Install system dependencies required for building Python packages.
RUN apk add --no-cache gcc musl-dev linux-headers

# Copy Pipfile and Pipfile.lock into the container.
COPY Pipfile Pipfile.lock ./

# Install pipenv and project dependencies in one step.
RUN pip install pipenv && \
    pipenv install --deploy --ignore-pipfile

# Copy the rest of the application source code into the container.
COPY . .

# Expose the port that the Flask app will run on.
EXPOSE 8000

# Set the environment variable for Flask (note: ensure app.py is your entry point).
ENV FLASK_APP=app.py

# Define the command to run your Flask app.
CMD ["pipenv", "run", "flask", "run", "--host=0.0.0.0", "--port=8000"]
