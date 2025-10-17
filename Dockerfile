# Dockerfile

# Use a specific, lightweight Python version as the base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching.
# This step will only be re-run if requirements.txt changes.
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project's source code into the container
COPY . .

# Download the necessary datasets into the image itself
# This makes the image fully self-contained
RUN python src/deco/download_datasets.py

# --- FIX FOR WINDOWS LINE ENDINGS ---
# This command strips the carriage return characters from the shell script
RUN sed -i 's/\r$//' run_all.sh

# Make the run script executable
RUN chmod +x run_all.sh

# Define the default command to run when the container starts.
# This will execute the entire experiment and plotting workflow.
CMD ["./run_all.sh"]