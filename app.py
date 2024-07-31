import streamlit as st
from svgpathtools import svg2paths2
import numpy as np
import cv2
from shapely.geometry import Polygon
import xml.etree.ElementTree as ET
from io import BytesIO
import base64
from scipy.optimize import leastsq
import math

# Function to determine the shape of the object
def detect_shape(contour):
    shape = "unidentified"
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    
    num_vertices = len(approx)
    
    if num_vertices == 3:
        shape = "triangle"
    elif num_vertices == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    elif 5 <= num_vertices <= 6:  # Handle pentagon and hexagon
        shape = "polygon"
    elif num_vertices > 6:  # Handle polygons with more than 6 sides
        shape = "polygon" 
    else:  # Check for circle and ellipse based on circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        shape = "circle" if circularity > 0.7 else "ellipse" 
    
    return shape

# Function to convert a path to a contour
def path_to_contour(path):
    contour = []
    for seg in path:
        for point in [seg.start, seg.end]:
            contour.append([int(point.real), int(point.imag)])
    return np.array(contour)

# Function to process SVG and detect shapes
def process_svg(file):
    file_data = file.read()
    
    paths, attributes, svg_attributes = svg2paths2(BytesIO(file_data))
    
    tree = ET.parse(BytesIO(file_data))
    root = tree.getroot()
    
    shape_counts = {
        "triangle": 0,
        "square": 0,
        "rectangle": 0,
        "polygon": 0,
        "circle": 0,
        "ellipse": 0,
        "unidentified": 0
    }
    
    for path in paths:
        contour = path_to_contour(path)
        
        if contour.size == 0:
            continue
        
        # Use a larger image to avoid cropping issues
        img = np.zeros((1000, 1000), dtype=np.uint8)
        cv2.fillPoly(img, [contour], 255)
        
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            shape = detect_shape(contour)
            if shape in shape_counts:
                shape_counts[shape] += 1
            else:
                shape_counts["unidentified"] += 1
    
    return shape_counts


# Function to convert SVG to base64
def svg_to_base64(svg_data):
    encoded = base64.b64encode(svg_data).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"

# Function to fit a circle using least squares
def fit_circle(XY):
    def objective(params):
        x0, y0, r = params
        return np.sqrt((XY[:, 0] - x0) ** 2 + (XY[:, 1] - y0) ** 2) - r

    # Initial guess
    x_mean = np.mean(XY[:, 0])
    y_mean = np.mean(XY[:, 1])
    r_guess = np.mean(np.sqrt((XY[:, 0] - x_mean) ** 2 + (XY[:, 1] - y_mean) ** 2))
    initial_params = [x_mean, y_mean, r_guess]
    
    result = leastsq(objective, initial_params)
    return result[0]

# Function to detect symmetry
def detect_symmetry(contours):
    def fit_line(points):
        # Fit a line and return its parameters (slope and intercept)
        A = np.vstack([points[:, 0], np.ones(len(points))]).T
        m, c = np.linalg.lstsq(A, points[:, 1], rcond=None)[0]
        return m, c
    
    def compute_symmetry_score(line_params, points):
        # Compute how symmetric the points are with respect to the line
        m, c = line_params
        distances = np.abs(m * points[:, 0] - points[:, 1] + c) / np.sqrt(m ** 2 + 1)
        return np.mean(distances)

    symmetric_lines = []
    for contour in contours:
        points = np.array(contour).reshape(-1, 2)
        if len(points) < 2:
            continue

        # Check for horizontal, vertical, and diagonal symmetry
        # 1. Horizontal Symmetry
        m_horizontal, c_horizontal = fit_line(points)
        score_horizontal = compute_symmetry_score((m_horizontal, c_horizontal), points)
        if score_horizontal < 10:  # Adjust threshold if needed
            symmetric_lines.append(("Horizontal", (m_horizontal, c_horizontal)))

        # 2. Vertical Symmetry
        m_vertical, c_vertical = fit_line(np.flip(points, axis=1))  # Flip points along the y-axis
        score_vertical = compute_symmetry_score((m_vertical, c_vertical), np.flip(points, axis=1))
        if score_vertical < 10:
            symmetric_lines.append(("Vertical", (m_vertical, c_vertical)))

        # 3. Diagonal Symmetry (45-degree angle)
        m_diagonal, c_diagonal = fit_line(points)
        # Calculate the angle of the diagonal line
        angle = math.degrees(math.atan(m_diagonal))
        if abs(angle - 45) < 5:  # Adjust the tolerance (5 degrees) if needed
            score_diagonal = compute_symmetry_score((m_diagonal, c_diagonal), points)
            if score_diagonal < 10:
                symmetric_lines.append(("Diagonal", (m_diagonal, c_diagonal)))

    return symmetric_lines

# Function to process symmetry detection
def symmetry_detection_page():
    st.header("Symmetry Detection")

    # Upload the image
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the image to create a binary mask
        ret, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find the contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Detect symmetry
        symmetric_lines = detect_symmetry(contours)

        # Display symmetric lines
        st.write(f"Detected {len(symmetric_lines)} lines of symmetry")
        for i, (line_type, (m, c)) in enumerate(symmetric_lines):
            st.write(f"Symmetric Line {i + 1}: {line_type} (y = {m:.2f}x + {c:.2f})")

# Function to process occlusion completion
def occlusion_completion_page():
    st.header("Occlusion Completion")

    # Upload the image
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the image to create a binary mask
        ret, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find the contours in the mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contours with largest area - the ring is the largest
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a mask for the largest contour
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)

        # Invert the mask
        inverted_mask = cv2.bitwise_not(contour_mask)

        # Apply the inverted mask to the original image
        completed_image = cv2.bitwise_and(image, image, mask=inverted_mask)

        # Occlusion Completion Logic
        for contour in contours:
            shape = detect_shape(contour)
            if shape == "circle":
                # Fit a circle to the contour
                x0, y0, r = fit_circle(np.array(contour).reshape(-1, 2))

                # Draw a filled circle on the completed image
                cv2.circle(completed_image, (int(x0), int(y0)), int(r), (0, 0, 0), -1)
            elif shape == "triangle":
                # Get the three vertices of the triangle (example: assuming they're in order)
                vertex1 = contour[0][0]
                vertex2 = contour[1][0]
                vertex3 = contour[2][0]

                # Fill the triangle on the completed image
                cv2.fillPoly(completed_image, [np.array([vertex1, vertex2, vertex3])], (0, 0, 0))
            # Add logic for other shapes you want to complete (e.g., rectangle, polygon)

        # Display the results
        st.image(completed_image, channels="BGR", caption="Completed Occlusion Image")

# Function to handle regularization page
def regularization_page():
    st.header("Regularization")

    # Upload the SVG file
    uploaded_file = st.file_uploader("Choose an SVG file", type="svg")

    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        
        # Convert SVG to base64
        svg_data = uploaded_file.read()
        base64_svg = svg_to_base64(svg_data)
        
        # Display the SVG file as an image
        st.image(base64_svg, use_column_width=True)
        
        # Reset the file pointer after displaying
        uploaded_file.seek(0)
        
        # Process the SVG file to detect shapes
        shape_counts = process_svg(uploaded_file)
        
        if shape_counts:
            st.write(f"Shape Counts: {shape_counts}")
            
                    
        else:
            st.write("No identifiable shapes found.")

def main():
    st.title("SVG Shape Detection")
    
    # Page selector
    page = st.sidebar.selectbox("Choose a Page", ["Regularization", "Occlusion Completion", "Symmetry Detection"])
    
    if page == "Occlusion Completion":
        occlusion_completion_page()
    elif page == "Regularization":
        regularization_page()
    elif page == "Symmetry Detection":
        symmetry_detection_page()

if __name__ == "__main__":
    main()