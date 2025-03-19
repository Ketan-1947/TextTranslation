import os
import easyocr
import cv2
import torch
from transformers import MarianMTModel, MarianTokenizer

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

# Initialize EasyOCR for English text detection
reader = easyocr.Reader(['en'], gpu=device == "cuda")

# Initialize MarianMT for English to Spanish translation
src_lang = "en"
tgt_lang = "es"
model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

def translate_text(text):
    # Skip empty text
    if not text.strip():
        return ""
    
    try:
        # Tokenize and translate
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        translated_ids = model.generate(**inputs, max_length=128)
        translation = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        return translation
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

# Capture video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Run OCR on the frame
    results = reader.readtext(frame)
    
    for (bbox, text, confidence) in results:
        # Only translate text with sufficient confidence
        if confidence > 0.5:
            translated_text = translate_text(text)
        else:
            translated_text = text  # Use original for low confidence
        
        # Get bounding box coordinates for the detected text
        top_left, top_right, bottom_right, bottom_left = bbox
        (x1, y1) = (int(top_left[0]), int(top_left[1]))
        (x2, y2) = (int(bottom_right[0]), int(bottom_right[1]))
        
        # Calculate text size to adjust the black box size
        (w, h), _ = cv2.getTextSize(translated_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Adjust the black box to fit the translated text
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 - h - 10), (0, 0, 0), -1)  # Black box
        
        # Overlay translated text inside the black box
        cv2.putText(frame, translated_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Display the frame with translated text overlay
    cv2.imshow("Translated Overlay", frame)
    key = cv2.waitKey(33)  # Wait 33ms for a keypress (~30fps)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()