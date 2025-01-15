# Use Python 3.10-slim as the base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /code

# Install build tools and pkg-config for h5py and patching
RUN apt-get update && \
    apt-get install -y build-essential gcc libhdf5-dev python3-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /code/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . /code/

# **Generate migrations**
RUN python manage.py makemigrations --no-input

# **Apply migrations**
RUN python manage.py migrate --no-input

# Expose the correct port
EXPOSE 8000

# Command to run your application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "saliencyslider.wsgi:application"]