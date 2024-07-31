# Use an official Python runtime as a parent image 
FROM python:3.10-slim 

# Set environment varibles 
ENV PYTHONDONTWRITEBYTECODE=1 
ENV PYTHONUNBUFFERED=1 

# Install system dependencies
RUN apt-get update && apt-get install -y \
    patch \
    && rm -rf /var/lib/apt/lists/*

# Set work directory 
WORKDIR /code 

#Install dependencies 
COPY requirements.txt /code/ 
RUN pip install --upgrade pip && pip install -r requirements.txt 

# Copy project 
COPY . /code/ 

# Copy and apply the patch to the library in site-packages
COPY changes.patch /tmp/changes.patch
RUN patch /usr/local/lib/python3.10/site-packages/vis/utils/utils.py < /tmp/changes.patch

# Run the application 
CMD ["gunicorn", "--bind", "0.0.0.0:80", "saliencyslider.wsgi:application"]
