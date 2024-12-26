import tkinter as tk
from tkinter import Label
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import pickle  # Assuming your stack_clf is saved as a pickle file

# Load your pre-trained model (stack_clf)
with open('mnist_model.pkl', 'rb') as f:
    stack_clf = pickle.load(f)

def save_and_predict():
    # Save the drawing from the canvas directly
    canvas.update()  # Ensure the canvas is updated
    ps_image = canvas.postscript(colormode="color")
    
    # Convert PostScript directly to an image
    from io import BytesIO
    img = Image.open(BytesIO(ps_image.encode("utf-8")))
    
    # Convert to grayscale, resize to 28x28, invert colors (MNIST-style)
    image = img.convert("L")
    image = image.resize((28, 28), Image.LANCZOS)
    image = ImageOps.invert(image)
    
    # Convert to a NumPy array
    img_array = np.array(image) / 255.0  # Normalize to range [0, 1]
    img_array = img_array.astype(np.float32)  # Ensure correct dtype
    
    img_array = np.expand_dims(img_array, axis=0)
    # Predict using the model
    print(img_array)
    prediction = stack_clf.predict(img_array)
    prediction_label.config(text=f"Predicted: {np.argmax(prediction)}")


def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, 200, 200), fill=(255))  # Clear drawing area

def paint(event):
    x1, y1 = (event.x - 5), (event.y - 5)
    x2, y2 = (event.x + 5), (event.y + 5)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=1)
    draw.line([x1, y1, x2, y2], fill="black", width=1)

# Set up the UI
root = tk.Tk()
root.title("MNIST Handwritten Digit Predictor")

# Canvas for drawing
canvas_frame = tk.Frame(root)
canvas_frame.pack()
canvas = tk.Canvas(canvas_frame, width=200, height=200, bg="white")
canvas.pack()

# Create a PIL Image and Draw object to store the drawing
image = Image.new("L", (200, 200), color=255)
draw = ImageDraw.Draw(image)

canvas.bind("<B1-Motion>", paint)

# Prediction label
prediction_label = Label(root, text="Draw a digit and click 'Predict'", font=("Helvetica", 16))
prediction_label.pack()

# Buttons
button_frame = tk.Frame(root)
button_frame.pack()

clear_button = tk.Button(button_frame, text="Clear", command=clear_canvas)
clear_button.pack(side="left", padx=5)

predict_button = tk.Button(button_frame, text="Predict", command=save_and_predict)
predict_button.pack(side="right", padx=5)

# Run the application
root.mainloop()
