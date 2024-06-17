# MiniCPM-Llama3: a GPT-4V Level Multimodal LLM

This repository contains a FastAPI application designed to process and analyze images.
The application downloads an image from a provided URL, processes it using a pre-trained model, and returns a detailed analysis in JSON format.

The model used for image processing is openbmb/MiniCPM-Llama3-V-2_5:
> https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5


## Features

- **Image Download and Processing**: Asynchronously downloads images from given URLs and processes them.
- **Detailed Image Analysis**: Provides a detailed JSON analysis of the image, including categories like people, animals, and environmental objects.
- **Asynchronous File Handling**: Efficiently handles file downloads and processing using aiohttp and aiofiles.
- **CORS Support**: Configured to support Cross-Origin Resource Sharing (CORS) for specified origins.


## Files

Inference using Huggingface transformers on NVIDIA GPUs

## Usage

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
``` 

## Monitoring Local GPU Usage

```
pip3 install --upgrade nvitop
```

## Requirements
- python 3.10
```
Pillow==10.1.0
torch==2.1.2
torchvision==0.16.2
transformers==4.40.0
sentencepiece==0.1.99
fastapi
uvicorn
pydantic
aiohttp
aiofiles
```

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/image-analysis-api.git
   cd image-analysis-api
   ```

2. Create a virtual environment and activate it:
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Start the FastAPI application:
   ```sh
   uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
   ```

2. Make a POST request to `/process_file` with a JSON payload containing the image URL:
   ```json
   {
     "url": "https://example.com/path/to/your/image.jpg"
   }
   ```

3. The API will return a JSON response with the image analysis.

### Docker Image

```
docker build -t minicpm .
docker run --gpus all -p 8000:8000 minicpm
```


## API Endpoints

### POST /process_file

Processes an image from a provided URL.

- **Request Body**:
  - `url`: The URL of the image to be processed.

- **Response**:
  - `message`: Confirmation message.
  - `results`: JSON analysis of the image.

