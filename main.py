import shutil
import os
import logging
import json
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from transformers import AutoTokenizer, AutoModel
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize the model and tokenizer once to avoid loading them multiple times
model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16).to(
    'cuda')
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval().to('cuda')


@app.post("/process_file")
async def process_file_endpoint(file: UploadFile, response: Response = None):
    if not file:
        return {"detail": "No file uploaded to FastAPI"}

    logger.info(f"Received file: {file.filename}")
    filename_without_extension, extension = os.path.splitext(file.filename)
    file_path = f"upload/{file.filename}"
    json_file_path = f"json/{filename_without_extension}.json"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        # Save the uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"File {file.filename} saved locally at {file_path}")

        logger.info("Starting Inference")
        result_json = await inference_minicpm(file_path)
        logger.info("Inference Done")

        with open(json_file_path, 'w') as json_file:
            json.dump(result_json, json_file)
        logger.info(f"Results saved at: {json_file_path}")

        response.headers["file_path"] = json_file_path
        response.headers["file_name"] = filename_without_extension
        return {
            "message": "JSON generated and saved successfully",
            "file_path": json_file_path,
            "file_name": filename_without_extension
        }

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()  # Get the detailed traceback
        print(traceback_str)
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download_json/{file_name}")
async def download_json(file_name: str):

    if ".." in file_name or "/" in file_name or "\\" in file_name:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    file_path = f"json/{file_name}.json"  # Construct the file path securely
    logger.info(f"Downloading file: {file_path}")

    if os.path.exists(file_path):
        return FileResponse(path=file_path, media_type='application/json')
    else:
        raise HTTPException(status_code=404, detail="File not found")


async def inference_minicpm(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
    prompt = prompt_model()
    msgs = [{'role': 'user', 'content': prompt}]
    res = model.chat(image=img, msgs=msgs, tokenizer=tokenizer, sampling=True, temperature=0.7)
    generated_text = "".join(res)
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
    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
