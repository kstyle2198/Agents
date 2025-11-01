# Use the official Python base image
FROM python:3.12.7-slim

# Set the working directory
WORKDIR frontend

# Copy the requirements.txt file (assuming you have one for your dependencies)
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . ./

# Expose the port that Streamlit runs on
EXPOSE 8501

# Set the environment variables for Streamlit
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true

# Run the Streamlit app
CMD ["streamlit", "run", "./app.py", "--server.port=8501", "--server.enableCORS=false"]
