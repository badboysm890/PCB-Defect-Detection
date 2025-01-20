#!/usr/bin/env python3

import os
import torch
import gradio as gr
from PIL import Image
import base64
from openai import OpenAI
from dotenv import load_dotenv
import warnings
import logging

# ---------------------------
# 0) Setup Logging and Warnings
# ---------------------------
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# ---------------------------
# 1) Load Environment Variables
# ---------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in .env file.")
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# 2) Load YOLOv5 Model
# ---------------------------
MODEL_PATH = "model/best.pt"
try:
    model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH)
    logger.info("YOLOv5 model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load YOLOv5 model: {e}")
    raise e

# ---------------------------
# 3) Helper Functions
# ---------------------------

def encode_image(image_path):
    """
    Encodes an image to a base64 string.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        logger.info(f"Image {image_path} encoded successfully.")
        return encoded_string
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return ""

def build_detection_summary(results):
    """
    Builds a summary of detections from YOLOv5 results.
    """
    detections = results.xyxy[0].tolist()  # [ [x1,y1,x2,y2,conf,class], ... ]
    class_names = results.names

    if not detections:
        return "No defects found."

    lines = []
    for i, det in enumerate(detections, start=1):
        conf = det[4]
        cls_index = int(det[5])
        conf_percent = conf * 100
        cls_name = class_names.get(cls_index, f"class_{cls_index}")
        lines.append(f"Defect #{i}: {cls_name} with {conf_percent:.2f}% confidence.")

    summary = "\n".join(lines)
    logger.info(f"Detection summary:\n{summary}")
    return summary

def process_assistant_response(response):
    """
    Processes the assistant's response from OpenAI to ensure it's a string.
    Embeds images using Markdown if present.
    """
    try:
        assistant_response = response.choices[0].message.content
        logger.info(f"Raw assistant response: {assistant_response}")

        # If the response is a list (as per user's example), process accordingly
        if isinstance(assistant_response, list):
            assistant_text = ""
            for item in assistant_response:
                if item.get('type') == 'text':
                    assistant_text += item.get('text', '') + "\n"
                elif item.get('type') == 'image_url':
                    img_url = item.get('image_url', {}).get('url', '')
                    if img_url:
                        # Embed image using Markdown for Gradio Chatbot
                        assistant_text += f"![Image]({img_url})\n"
            logger.info(f"Processed assistant text: {assistant_text}")
            return assistant_text.strip()
        elif isinstance(assistant_response, str):
            logger.info(f"Assistant response is a string: {assistant_response}")
            return assistant_response
        else:
            # Fallback for unexpected formats
            logger.warning("Assistant response is in an unexpected format.")
            return str(assistant_response)
    except Exception as e:
        logger.error(f"Error processing assistant response: {e}")
        return "Error processing the assistant's response."

# ---------------------------
# 4) Chat Functions
# ---------------------------

def detect_and_init_chat(image_pil):
    """
    1) Runs YOLOv5 on the input image.
    2) Creates an annotated image with bounding boxes.
    3) Builds a text summary of each defect + confidence.
    4) Saves the annotated image to disk.
    5) Calls OpenAI with the annotated image and detection summary.
    6) Returns the annotated image for display and the conversation state.
    """
    try:
        # YOLO detection
        results = model(image_pil)
        logger.info("YOLOv5 detection completed.")

        # Annotate in memory
        annotated_array = results.render()[0]
        annotated_image = Image.fromarray(annotated_array)

        # Build detection summary
        detection_summary = build_detection_summary(results)

        # Save the annotated image
        annotated_path = "temp_annotated.jpg"
        annotated_image.save(annotated_path)
        logger.info(f"Annotated image saved to {annotated_path}.")

        # Encode the annotated image to base64
        base64_image = encode_image(annotated_path)

        # Prepare messages as per your example
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Here is a PCB image with YOLO-detected bounding boxes. Below is the detection summary:\n"
                            f"{detection_summary}\n\n"
                            "Please analyze the image, comment on the defects, and describe any anomalies."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ]

        # Call OpenAI's Chat Completion API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Ensure this model supports vision
            messages=messages,
            max_tokens=500,  # Adjust as needed
        )
        logger.info("OpenAI ChatCompletion API called successfully.")

        # Process assistant's response
        assistant_text = process_assistant_response(response)

        # Initialize conversation state
        conversation_state = {
            "history": [["", assistant_text]],  # [user_message, assistant_message]
            "image_path": annotated_path,
        }

        return annotated_image, conversation_state

    except Exception as e:
        logger.error(f"Error in detect_and_init_chat: {e}")
        # Return the original image and an error state
        return image_pil, {
            "history": [["", f"Error during detection and chat initialization: {e}"]],
            "image_path": None,
        }

def continue_chat(user_message, conversation_state):
    """
    Handles user follow-up messages by:
    - Reconstructing the conversation history.
    - Reattaching the annotated image.
    - Appending the new user message.
    - Calling OpenAI to get the assistant's reply.
    - Updating the conversation state.
    """
    try:
        history = conversation_state.get("history", [])
        annotated_path = conversation_state.get("image_path", "")

        if not annotated_path:
            logger.error("Annotated image path not found in conversation state.")
            return "Error: Annotated image not found.", conversation_state

        # Encode the annotated image to base64
        base64_image = encode_image(annotated_path)

        # Reconstruct messages for OpenAI
        messages = []
        for user_text, assistant_text in history:
            if user_text:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                })
            if assistant_text:
                messages.append({
                    "role": "assistant",
                    "content": assistant_text
                })

        # Add the current user message
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_message,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        })

        # Call OpenAI's Chat Completion API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Ensure this model supports vision
            messages=messages,
            max_tokens=500,  # Adjust as needed
        )
        logger.info("OpenAI ChatCompletion API called successfully for follow-up.")

        # Process assistant's response
        assistant_text = process_assistant_response(response)

        # Update conversation history
        history.append([user_message, assistant_text])
        conversation_state["history"] = history

        return "", conversation_state

    except Exception as e:
        logger.error(f"Error in continue_chat: {e}")
        # Return the existing state with an error message
        history = conversation_state.get("history", [])
        history.append([user_message, f"Error during chat continuation: {e}"])
        conversation_state["history"] = history
        return "", conversation_state

def update_chat(state):
    """
    Formats the conversation history for Gradio's Chatbot component.
    Ensures each message is a list of two strings.
    """
    try:
        history = state.get("history", [])
        if not history:
            return []
        formatted_history = []
        for pair in history:
            if len(pair) != 2:
                logger.warning(f"Invalid history pair: {pair}")
                # Skip or handle appropriately
                continue
            user, assistant = pair
            formatted_history.append([user, assistant])
        logger.info(f"Formatted history for Chatbot: {formatted_history}")
        return formatted_history
    except Exception as e:
        logger.error(f"Error in update_chat: {e}")
        return []

# ---------------------------
# 5) Build Gradio Interface
# ---------------------------
def build_interface():
    with gr.Blocks() as demo:

        gr.Markdown("# PCB Defect Detection and Analysis")

        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload PCB Image")
            detect_button = gr.Button("Detect & Describe")

        annotated_output = gr.Image(type="pil", label="YOLO-annotated Image")
        chatbot = gr.Chatbot(label="Chat with Vision Model")

        # Conversation state => { "history": [...], "image_path": ... }
        conversation_state = gr.State({"history": [], "image_path": None})

        # Textbox for user questions
        user_input = gr.Textbox(label="Follow-up question...", placeholder="Ask a question about the defects...")

        # Detect/Describe => detect_and_init_chat
        detect_button.click(
            fn=detect_and_init_chat,
            inputs=image_input,
            outputs=[annotated_output, conversation_state]
        )

        # Whenever conversation_state changes => show it in chatbot
        conversation_state.change(
            fn=update_chat,
            inputs=conversation_state,
            outputs=chatbot
        )

        # Follow-up => continue_chat
        user_input.submit(
            fn=continue_chat,
            inputs=[user_input, conversation_state],
            outputs=[user_input, conversation_state]
        )

        # Update chatbot after follow-up
        conversation_state.change(
            fn=update_chat,
            inputs=conversation_state,
            outputs=chatbot
        )

    return demo

# ---------------------------
# 6) Launch the Interface
# ---------------------------
if __name__ == "__main__":
    iface = build_interface()
    iface.launch(
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=7860,       # Port to listen on inside the container
        share=True              # Disable public share links
    )
