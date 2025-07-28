FROM --platform=linux/amd64 python:3.10-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
# This is done first to leverage Docker's build cache
COPY requirements.txt .

# Install the Python dependencies specified in requirements.txt
# --no-cache-dir ensures that pip doesn't store cache, keeping the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy your main PDF processing script into the container
COPY process_pdfs.py .

# Command to run your application when the container starts
CMD ["python", "process_pdfs.py"]