# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir ./debate-for-ai-alignment
RUN pip install --no-cache-dir -r ./debate-ui/requirements.txt

# Make port 8050 available to the world outside this container
EXPOSE 8050

# Change the working directory to the app directory
WORKDIR /app/debate-ui

# Run app.py when the container launches
CMD ["gunicorn", "-b", "0.0.0.0:8050", "src.app:server"]