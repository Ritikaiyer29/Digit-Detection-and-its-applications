import streamlit as st
import numpy as np
import cv2
import pickle
from streamlit_drawable_canvas import st_canvas

# Load the trained model using pickle
with open('C:\\Users\\RITIKA\\Desktop\\2nd year\\sem 4\\Sodoku project\\random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to preprocess the image
def preprocess_image(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img_resized = cv2.resize(img_gray, (28, 28))  # Resize to model input size
    img_flattened = img_resized.flatten()  # Flatten the image
    img = np.array(img_flattened) / 255.0  # Normalize pixel values
    return img.reshape(1, -1)  # Reshape for model input


def predict_digit(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    return prediction[0]  # Return the predicted digit

# Streamlit app
def main():
    st.title('Digit Detection App')
    st.write('Draw a digit in the canvas below and click the Predict button!')

    # Create a canvas to draw digits
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=20,
        stroke_color='#ffffff',
        background_color='#000000',
        width=200,
        height=200,
        drawing_mode="freedraw",
        key="canvas")

    # Button to predict
    if st.button('Predict'):
        if canvas_result.image_data is not None:
            # Convert canvas image to OpenCV format
            image = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2BGR)
            digit = predict_digit(image)
            st.write('Predicted Digit:', digit)

if __name__ == '__main__':
    main()
