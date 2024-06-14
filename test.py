import argparse
import json
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from transformers import AutoModel, AutoTokenizer

from utils import build_logger

# Load environment variables from .env.local file
load_dotenv('.env.local')

logger = build_logger("controller", "Logs/controller.log")
GO_SERVER_URL = os.getenv('GO_SERVER_URL', 'http://localhost:8080')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:21001", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.post("/process_file")
async def process_file_endpoint(file: UploadFile = File(...), response: Response = None):
    contents = await file.read()
    logger.info(f"Received file: {file.filename}")

    filename_without_extension, extension = os.path.splitext(file.filename)
    file_path = f"MiniCPM/files/{file.filename}"
    json_file_path = f"MiniCPM/files/{filename_without_extension}.json"

    logger.info(f"Saving Original file at: {file_path}")
    with open(file_path, "wb") as f:
        f.write(contents)

    logger.info(f"File saved at: {file_path}")

    # image = Image.open(save_path).convert('RGB')
    # prompt = prompt_model()
    # result = await mini_cpm(image, prompt)

    result = {
        "Global Answer": "The image captures a moment of religious study within a community setting, likely a synagogue. The individuals are engaged in reading or studying religious texts, which is a common practice during prayer services and religious gatherings. The presence of the yarmulke and tallit suggests that this is a Jewish context. The stained glass window adds an element of cultural significance, often found in places of worship, symbolizing light and spirituality. The overall atmosphere appears to be one of reverence and communal participation in religious observance.",
        "people": {
            "count": 5,
            "details": [
                {"gender": "male", "age": "young child", "clothing": "white shirt with blue stripes",
                 "activity": "reading"},
                {"gender": "male", "age": "adult", "clothing": "white shirt with blue stripes",
                 "activity": "reading"},
                {"gender": "female", "age": "adult", "clothing": "black hat and jacket", "activity": "reading"},
                {"gender": "male", "age": "adult", "clothing": "black hat and jacket", "activity": "reading"},
                {"gender": "male", "age": "adult", "clothing": "black hat and jacket", "activity": "praying"}
            ]
        },
        "Environmental Objects": {
            "Artificial Structures": "stained glass window",
            "Natural Elements": "none",
            "Lights": "soft natural light filtering through the window",
            "Color Dominance": "multi-colored stained glass with prominent reds, blues, and yellows"
        }
    }

    try:
        with open(json_file_path, 'w') as file:
            json.dump(result, file)
        response.headers["file_path"] = json_file_path
        response.headers["file_name"] = filename_without_extension
        return {"message": f"JSON generated and saved successfully at {json_file_path}", "file_path": json_file_path,
                "file_name": filename_without_extension}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download_json/{file_path:path}")
async def download_json(file_path: str):
    logger.info(f"Downloading file: {file_path}")
    try:
        return FileResponse(path=file_path, media_type='application/json')
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")


def mini_cpm(img, prompt: str):
    # Initialize the model and tokenizer once
    model, tokenizer = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4',
                                                 trust_remote_code=True), AutoTokenizer.from_pretrained(
        'openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
    model.eval()

    msgs = [{'role': 'user', 'content': prompt}]
    res = model.chat(image=img, msgs=msgs, tokenizer=tokenizer, sampling=True, temperature=0.7)
    generated_text = "".join(res)  # Assuming this results in a string
    result_json = {
        "analysis": generated_text
    }
    return result_json


def prompt_model():
    prompt = """
        Analyze the provided image and produce a detailed description in JSON format. Focus on identifying and categorizing all significant elements within the image. Generate categories dynamically based on the content observed, and detail each category with precision. Only include categories with visible, relevant details and ensure each description is specific to the context of the image.

        Instructions for Comprehensive Analysis:
        Identify All Entities:

        People: Describe any people in the image, including their activities, expressions, and interactions with the environment or other beings.
        Animals: Identify and describe all animals, noting their species, count, behavior, and interaction with humans or the environment.
        Artificial Structures and Natural Elements: Note any significant man-made or natural elements that contribute to the scene's context.
        Environmental and Contextual Details:

        Lighting and Color: Describe the lighting and dominant colors if they significantly impact the scene.
        Setting Description: Provide context such as the location type (e.g., zoo, wildlife park) if identifiable from the image.
        Required JSON Structure:
        Ensure that each relevant category is filled with comprehensive details and dynamically create or omit categories based on the image content. Here's how you must structure your JSON, with an example:
        {
        "Global Answer": "The image captures a candid moment between two individuals, likely a mother and her child, in a fast-food restaurant. The child is holding a balloon, which is often associated with joy or a special occasion, possibly indicating they are celebrating something. The adult's playful gesture with the peace sign could be interpreted as a lighthearted interaction or a way to engage with the camera. The presence of food on the table suggests they are in the midst of a meal. The background provides context that this is a public space designed for dining, indicated by the restaurant's branding and menu display. The attire of the individuals is casual, fitting for a relaxed dining experience. Overall, the image conveys a sense of everyday life and family bonding.",
        "people": {
        "count": 3,
        "details": [
        {"gender": "female", "origin": "Asian", "clothing": "red dress", "expression": "smiling", "features": "sunglasses"},
        {"gender": "male", "origin": "African", "clothing": "blue suit", "expression": "serious", "features": "none"},
        {"gender": "female", "origin": "European", "clothing": "green shirt and jeans", "expression": "laughing", "features": "hat"}
        ]
        },
        "animals": {
        "count": "specify number",
        "details": [
        {
        "species": "name the species",
        "behavior": "describe behavior",
        "interaction": "note any interactions with humans"
        }
        ]
        },
        "Environmental Objects": {
        "Artificial Structures": "describe any significant structures",
        "Natural Elements": "note significant natural elements",
        "Lights": "describe the lighting",
        "Color Dominance": "mention dominant colors"
        }
        }
        Additional Instructions to the Model:
        Be specific and direct in identifying and counting all significant beings and objects.
        If multiple entities of the same type are visible (like two tigers), ensure to note each one and describe their individual characteristics and actions.
        Avoid generic descriptions; instead, focus on providing details that offer clear insights into the scene.
        Religious Sensitivity: If religious symbols or items are visible, refer to them only if they are essential to the context of the scene, and always use general terms such as 'cultural symbols' without specifying the religion.
    """
    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21001)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
