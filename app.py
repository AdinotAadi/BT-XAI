import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# Load your pre-trained model
model = tf.keras.models.load_model('model/brain_tumor_model.h5')

# Specify the names of the last convolutional layer and classifier layer
last_conv_layer_name = 'conv2d_97'
classifier_layer_names = ['dense_9']  # Adjust as needed for your model

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, classifier_layer_names):
    # Create a model to extract the gradients
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Record the gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([img_array]))
        
        # Get the class index with the highest probability
        class_index = tf.argmax(predictions[0]).numpy()
        
        # Use the class index to find the loss
        loss = predictions[0][class_index]
    
    # Get the gradients of the last conv layer
    grads = tape.gradient(loss, conv_outputs)
    
    # Compute the mean of the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Get the output of the last conv layer
    conv_outputs = conv_outputs[0]
    
    # Weight the output of the conv layer with the pooled grads
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    
    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap)  # Normalize to [0, 1]
    
    return heatmap

# Streamlit app
st.title("Brain Tumor Classification with Explainable AI")

# Allow multiple file uploads
uploaded_files = st.file_uploader("Choose images...", type="jpg", accept_multiple_files=True)

if uploaded_files is not None and len(uploaded_files) > 0:
    # Create lists to hold the original images and results
    images = []
    heatmaps = []
    predictions = []
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

    for uploaded_file in uploaded_files:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        image = image.resize((224, 224))  # Resize according to your model input size
        img_array = np.array(image) / 255.0  # Normalize the image
        images.append(image)

        # Make predictions
        preds = model.predict(np.array([img_array]))
        predicted_class = class_names[np.argmax(preds)]
        predictions.append((predicted_class, preds))

        # Generate Grad-CAM
        heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name, classifier_layer_names)
        heatmaps.append(heatmap)

    # Display the results
    for idx, (img, heatmap, (pred_class, pred_probs)) in enumerate(zip(images, heatmaps, predictions)):
        # Resize heatmap to the original image size
        heatmap_resized = cv2.resize(heatmap, (img.width, img.height))
        heatmap_resized = np.uint8(255 * heatmap_resized)

        # Apply colormap
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        superimposed_img = heatmap_color * 0.4 + np.array(img)

        # Create a container for each image and Grad-CAM
        with st.container():
            cols = st.columns(2)
            with cols[0]:
                st.image(img, caption='Original MRI Image', use_column_width=True)
                if pred_class == 'No Tumor':
                    st.markdown(f"<h2 style='color: green; font-weight: bold;'>{pred_class}</h2>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='color: red; font-weight: bold;'>{pred_class}</h2>", unsafe_allow_html=True)
            with cols[1]:
                st.image(superimposed_img.astype(np.uint8), caption='Grad-CAM', use_column_width=True)

else:
    st.write("Please upload images to see predictions and Grad-CAM visualizations.")