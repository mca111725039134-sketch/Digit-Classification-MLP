import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import joblib

# Load trained model
model = joblib.load("model/digit_model.pkl")

# Create window
root = tk.Tk()
root.title("Digit Classification using MLP")

canvas_width = 280
canvas_height = 280

canvas = tk.Canvas(
    root,
    width=canvas_width,
    height=canvas_height,
    bg="white",
    cursor="cross"
)
canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

# Create blank image
image = Image.new("L", (canvas_width, canvas_height), color=255)
draw = ImageDraw.Draw(image)

# Labels
prediction_label = Label(root, text="Prediction : ", font=("Arial", 14))
prediction_label.grid(row=1, column=0, sticky="w", padx=10)

confidence_label = Label(root, text="Confidence : ", font=("Arial", 14))
confidence_label.grid(row=2, column=0, sticky="w", padx=10)

# Draw on canvas
def paint(event):
    x1 = event.x - 8
    y1 = event.y - 8
    x2 = event.x + 8
    y2 = event.y + 8

    canvas.create_oval(x1, y1, x2, y2,
                       fill="black",
                       outline="black")

    draw.ellipse([x1, y1, x2, y2],
                 fill=0)

canvas.bind("<B1-Motion>", paint)
# Clear canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill=255)
    prediction_label.config(text="Prediction : ")
    confidence_label.config(text="Confidence : ")

# Predict digit
def predict_digit():
    # Resize to 28x28 (same size as MNIST)
    img = image.resize((28, 28))
    img = ImageOps.invert(img)

    # Convert image to numpy array
    img_array = np.array(img).reshape(1, 784)

    print("Image dtype:", img_array.dtype)
    print("Unique values:", np.unique(img_array)[:20])
    print("Min:", img_array.min())
    print("Max:", img_array.max())

    # Predict
    prediction = model.predict(img_array)[0]

    # Confidence
    probabilities = model.predict_proba(img_array)
    confidence = np.max(probabilities) * 100

    prediction_label.config(text=f"Prediction : {prediction}")
    confidence_label.config(text=f"Confidence : {confidence:.2f}%")

# Buttons
predict_button = Button(
    root,
    text="Predict",
    command=predict_digit,
    width=12,
    bg="lightgreen"
)
predict_button.grid(row=3, column=0, pady=10)

clear_button = Button(
    root,
    text="Clear",
    command=clear_canvas,
    width=12,
    bg="tomato"
)
clear_button.grid(row=3, column=1, pady=10)

# Start GUI
root.mainloop()
