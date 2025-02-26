import yt_dlp
import os
import json
import re
import concurrent.futures

# Load the uploaded JSON file
with open('ontology.json', 'r') as f:
    json_data = json.load(f)

# Directory to save downloaded audio files
download_directory = "audiosets/ontology"
if not os.path.exists(download_directory):
    os.makedirs(download_directory)

# Function to download audio from YouTube URL
def download_audio_from_url(url, start_time, end_time, file_name):
    try:
        # Check if the file already exists
        file_path = os.path.join(download_directory, file_name)
        if os.path.exists(file_path):
            print(f"File {file_name} already exists, skipping download.")
            return  # Skip download if the file already exists

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': file_path,
            'noplaylist': True,  # Avoid downloading entire playlists
            'postprocessor_args': [
                '-ss', str(start_time),
                '-to', str(end_time)
            ],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"Downloaded audio for {file_name} from {url}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

# Helper function to extract start and end times from URL
def extract_times(url):
    match = re.search(r"start=(\d+)&end=(\d+)", url)
    if match:
        start_time = int(match.group(1))
        end_time = int(match.group(2))
        return start_time, end_time
    return None, None

# Function to handle downloading for each category and URL
def download_audio_for_category(category, idx, url):
    start_time, end_time = extract_times(url)
    if start_time is not None and end_time is not None:
        # Generate a file name based on the category and index
        file_name = f"{category['name']}_{idx}.mp3"
        # Download audio for the current URL
        download_audio_from_url(url, start_time, end_time, file_name)
    else:
        print(f"Could not extract time parameters from URL: {url}")

# Using ThreadPoolExecutor to download audios in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for category in json_data:
        if "positive_examples" in category:
            for idx, url in enumerate(category["positive_examples"]):
                futures.append(executor.submit(download_audio_for_category, category, idx, url))

    # Wait for all the threads to finish
    concurrent.futures.wait(futures)

print("Download complete.")
