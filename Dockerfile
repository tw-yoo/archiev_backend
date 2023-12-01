# Use an official Python runtime as a parent image
FROM --platform=linux/amd64 python:3.10-slim as build

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

RUN apt-get update && apt-get install -y gcc g++ make libffi-dev python3-dev
RUN pip install --upgrade pip setuptools wheel

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]