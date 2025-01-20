#!/usr/bin/env python3

import os
import torch
import gradio as gr
from PIL import Image
import ollama

# ---------------------------
# 1) Load YOLOv5 Model
# ---------------------------
MODEL_PATH = "model/best.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH)

# ---------------------------
# 2) Per-Defect Summary
# ---------------------------
def build_detection_summary(results):
    """
    Returns a multi-line string describing each detection:
       "Defect #1: <class_name> with 94.32% confidence."
    If no objects, returns "No defects found."
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
    
    return "\n".join(lines)

# ---------------------------
# 3) Detect & Initialize Chat
# ---------------------------
def detect_and_init_chat(image_pil):
    """
    1) Runs YOLOv5 on the input image.
    2) Creates an in-memory annotated image (with bounding boxes).
    3) Builds a text summary of each defect + confidence.
    4) Saves the *annotated* image to disk so Ollama sees the marked image.
    5) Calls Ollama with that annotated image + the detection summary.
    6) Returns (annotated_image_for_display, conversation_state).
    """
    # YOLO detection
    results = model(image_pil)

    # Annotate in memory
    annotated_array = results.render()[0]
    annotated_image = Image.fromarray(annotated_array)

    # Summaries
    detection_summary = build_detection_summary(results)

    # Save the annotated image for the vision model
    annotated_path = "temp_annotated.jpg"
    annotated_image.save(annotated_path)

    # Prompt text includes the detection summary
    prompt_text = (
        "Here is a PCB image with YOLO-detected bounding boxes. Below is the detection summary:\n"
        f"{detection_summary}\n\n"
        "Please analyze the image, comment on the defects, and describe any anomalies."
    )

    # Call Ollama with the annotated image
    response = ollama.chat(
        model="llama3.2-vision",
        messages=[
            {
                "role": "user",
                "content": prompt_text,
                "images": [annotated_path],  # attach annotated image
            }
        ]
    )

    # Extract text from Ollama's response
    assistant_text = response.message.content if hasattr(response, "message") else str(response)

    # Conversation state (dict) => store entire history + the path to annotated image
    conversation_state = {
        "history": [(None, assistant_text)],
        "image_path": annotated_path  # we keep the annotated image path for subsequent turns
    }

    return annotated_image, conversation_state

# ---------------------------
# 4) Continue Chat
# ---------------------------
def continue_chat(user_message, conversation_state):
    """
    On each user follow-up:
      - Reconstruct conversation in Ollama format
      - Reattach the *annotated* image
      - Append new user message
      - Return new assistant reply
    """
    history = conversation_state["history"]
    annotated_path = conversation_state["image_path"]

    # Build Ollama message history
    messages = []
    for user_text, assistant_text in history:
        if user_text:
            messages.append({
                "role": "user",
                "content": user_text,
                "images": [annotated_path],
            })
        if assistant_text:
            messages.append({
                "role": "assistant",
                "content": assistant_text
            })

    # Add current user message, also attach annotated image
    messages.append({
        "role": "user",
        "content": user_message,
        "images": [annotated_path],
    })

    # Call Ollama
    response = ollama.chat(
        model="llama3.2-vision",
        messages=messages
    )
    assistant_text = response.message.content if hasattr(response, "message") else str(response)

    # Update conversation history
    history.append((user_message, assistant_text))
    conversation_state["history"] = history

    return "", conversation_state

# ---------------------------
# 5) Build Gradio UI
# ---------------------------
def build_interface():
    with gr.Blocks() as demo:

        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload PCB Image")
            detect_button = gr.Button("Detect & Describe")

        annotated_output = gr.Image(type="pil", label="YOLO-annotated Image")
        chatbot = gr.Chatbot(label="Chat with Vision Model")

        # Conversation state => { "history": [...], "image_path": ... }
        conversation_state = gr.State({"history": [], "image_path": None})

        # Textbox for user questions
        user_input = gr.Textbox(label="Follow-up question...")

        # Detect/Describe => detect_and_init_chat
        detect_button.click(
            fn=detect_and_init_chat,
            inputs=image_input,
            outputs=[annotated_output, conversation_state]
        )

        # Whenever conversation_state changes => show it in chatbot
        conversation_state.change(
            lambda s: s["history"],
            conversation_state,
            chatbot
        )

        # Follow-up => continue_chat
        user_input.submit(
            fn=continue_chat,
            inputs=[user_input, conversation_state],
            outputs=[user_input, conversation_state]
        )

        conversation_state.change(
            lambda s: s["history"],
            conversation_state,
            chatbot
        )

    return demo

# ---------------------------
# 6) Launch
# ---------------------------
if __name__ == "__main__":
    iface = build_interface()
    iface.launch()
