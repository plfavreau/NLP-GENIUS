FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the dataset into a data directory in the container
COPY dataset/song_lyrics.csv dataset/

# Copy the entire transformers directory
COPY transformers/ transformers/

# Install Python packages from the requirements file
RUN pip install --upgrade pip && \
    pip install -r transformers/requirements.txt --no-cache-dir


WORKDIR /app/transformers

CMD ["python3", "title_to_lyrics_transformer.py"]
