# Use a Python slim image for a smaller base size
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Pre-download the Sentence-Transformer Model ---
# This step downloads the model weights into the Docker image
# It will be cached in /root/.cache/huggingface/transformers/ or similar
# The `SentenceTransformer` class automatically looks for models here
# or within the model_name itself if it's a local path.
ARG MODEL_NAME="all-MiniLM-L6-v2"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')"

# Copy all your application code
COPY . .

# Set the entry point for the application.
# The actual execution command will be given by the challenge runner,
# but for local testing, this is what you'd use.
CMD ["python", "challenge1b_processor.py"]