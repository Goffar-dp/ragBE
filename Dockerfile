# Use a lightweight Python image
FROM python:3.9-slim-buster

# Install system dependencies for Poppler, Tesseract, and sqlite3-dev
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-ben \
    libsqlite3-dev \
    # Clean up APT cache to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000 to avoid permission issues
RUN useradd -m -u 1000 user
# Switch to the "user" user
USER user
# Set home to the user's home directory and add local bin to PATH
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy your requirements.txt file into the container, owned by the new user
COPY --chown=user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container, owned by the new user
COPY --chown=user . .

# Expose the port Streamlit runs on (default for Streamlit)
EXPOSE 8501

# Command to run your Streamlit application
# --server.port 8501 is the port exposed by the container
# --server.address 0.0.0.0 makes it accessible from outside the container
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
