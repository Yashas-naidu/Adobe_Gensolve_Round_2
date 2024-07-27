# Adobe_Gensolve_Round_2

To solve the problems outlined in the document, you'll need to follow a series of steps that involve reading the input data, processing it to identify and regularize shapes, and then comparing your output to the provided solutions. Here's a step-by-step approach:

### Step-by-Step Guide

1. **Read the Input Data**:
   - Use the provided CSV files to read the raw data points for the curves.
   - Each CSV file represents a set of points that form polylines (curves).

2. **Regularize Curves**:
   - Identify and straighten common shapes in the curves, such as straight lines, circles, ellipses, rectangles, polygons, and star shapes.
   - You can use mathematical methods to detect these shapes based on the points' properties.

3. **Explore Symmetry**:
   - Check for reflection symmetries in closed shapes.
   - Identify lines of symmetry where the shape can be divided into mirrored halves.

4. **Complete Incomplete Curves**:
   - Develop algorithms to fill in gaps in curves caused by occlusion.
   - Handle different levels of shape occlusion, from fully contained to partially contained and disconnected shapes.

5. **Compare with Expected Results**:
   - Use the provided solution files (`*_sol.csv` and `*_sol.svg`) to compare your output.
   - Evaluate the accuracy of your results by visualizing and comparing them to the expected solutions.

## Deep dive more 

#### 1. Reading CSV Files

#### 2. Identifying Shapes

Next, you'll need to develop algorithms to identify and regularize shapes. For example, to detect circles, you can use the least squares method to fit a circle to a set of points.

#### 3. Exploring Symmetry

To find lines of symmetry, you can compare points on either side of a potential symmetry line.

#### 4. Completing Curves

For completing curves, you can use interpolation methods to fill in gaps. For example, spline interpolation can be used to smoothly connect points.

### Using Provided Solution Files

After processing the curves, compare your results to the provided solution files to ensure accuracy.

### Tools and Libraries

- **NumPy**: For numerical operations and handling arrays.
- **Matplotlib**: For plotting and visualizing the curves.
- **SciPy**: For advanced mathematical operations like curve fitting and interpolation.
