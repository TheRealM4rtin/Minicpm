import os
import json
from PIL import Image
import torch
import logging

from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_file(file_path):
    try:

        filename_without_extension, extension = os.path.splitext(file_path)
        json_file_path = f"json/{filename_without_extension}.json"

        # Initialize the model and tokenizer once
        model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True,
                                          torch_dtype=torch.float16)
        model = model.to(device='cuda')

        tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
        model.eval()

        img = Image.open(file_path).convert('RGB')

        prompt = prompt_model()

        msgs = [{'role': 'user', 'content': prompt}]

        res = model.chat(image=img, msgs=msgs, tokenizer=tokenizer, sampling=True, temperature=0.7)

        print(res)

        generated_text = "".join(res)
        result_json = {
            "analysis": generated_text
        }

        # Save the result to a JSON file
        with open(json_file_path, 'w') as file:
            json.dump(result_json, file)

        logger.info(f"JSON generated and saved successfully at {json_file_path}")
        return json_file_path

    except Exception as e:
        logger.error("Failed to process the file due to an error: " + str(e))
        return None


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
    directory_path = "media/"
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            output = process_file(file_path)
            if output:
                logger.info(f"Process completed. JSON saved at {output}")
            else:
                logger.info("Process failed for file: " + filename)
