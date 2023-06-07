import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
import cv2
from PIL import Image   
import dataiku
import matplotlib.pyplot as plt
import io
import base64

#Preprocessing Lib Start

def random_color(C):
    # C is an integer that represents the number of colors to output
    # The function returns a list of C random and distinct RGB colors
    # The colors are not too white (i.e. the sum of RGB values is less than 600)
    colors = [] # an empty list to store the colors

    while len(colors) < C: # loop until we have C colors
        r = np.random.randint(0, 255) # generate a random red value
        g = np.random.randint(0, 255) # generate a random green value
        b = np.random.randint(0, 255) # generate a random blue value
        color = (r, g, b) # create a tuple for the color
        if sum(color) < 600 and color not in colors: # check if the color is not too white and not already in the list
            colors.append(color) # add the color to the list
    return colors # return the list of colors

def rgb_to_hex(rgb):
    """Converts a tuple of RGB values to a hexadecimal color code.

    Parameters:
        rgb (tuple): A tuple of three integers in the range 0-255 representing the red, green and blue components of the color.

    Returns:
        str: A string of six hexadecimal digits representing the color code.

    Example:
        >>> rgb_to_hex((255, 0, 0))
        'FF0000'
    """
    # Check if the input is a valid RGB tuple
    if not isinstance(rgb, tuple) or len(rgb) != 3 or not all(0 <= x <= 255 for x in rgb):
        raise ValueError("Invalid RGB input")

    # Convert each component to a two-digit hexadecimal string
    hex_components = [format(x, "02X") for x in rgb]

    # Join the components and return the result
    return "".join(hex_components)

def generate_tuples(N):
    result = []
    for i in range(N):
        result.append(((255-i)/255, 0, 0))
    return result

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def resize(img,k):
    return cv2.resize(img, dsize=(int(img.shape[1]*k), int(img.shape[0]*k)), interpolation=cv2.INTER_NEAREST)


def crop_image_v2(image, crop_size, pixel_value):
  # Get the shape of the image
  height, width, N = image.shape
  # Get the crop height and width
  crop_height, crop_width = crop_size
  # Calculate the number of crops along each dimension
  num_crops_h = height // crop_height + (height % crop_height != 0)
  num_crops_w = width // crop_width + (width % crop_width != 0)
  # Calculate the total number of crops
  num_crops = num_crops_h * num_crops_w
  # Initialize the output array with the padding value
  output = np.full((num_crops, crop_height, crop_width, N), pixel_value)
  # Loop over the crops and copy the image data
  for i in range(num_crops_h):
    for j in range(num_crops_w):
      # Get the index of the current crop
      k = i * num_crops_w + j
      # Get the start and end coordinates of the image region to copy
      start_y = i * crop_height
      end_y = min((i + 1) * crop_height, height)
      start_x = j * crop_width
      end_x = min((j + 1) * crop_width, width)
      # Get the start and end coordinates of the output region to paste
      out_start_y = 0
      out_end_y = end_y - start_y
      out_start_x = 0
      out_end_x = end_x - start_x
      # Copy the image data to the output array
      output[k, out_start_y:out_end_y, out_start_x:out_end_x, :] = image[start_y:end_y, start_x:end_x, :]
  # Return the output array and the number of crops along each dimension
  return output, (num_crops_h, num_crops_w)

def uncrop_image_v2(cropped_images, num_crops):
  # Get the shape of the cropped images
  num_crops_total, crop_height, crop_width, N = cropped_images.shape
  # Get the number of crops along each dimension
  num_crops_h, num_crops_w = num_crops
  # Calculate the height and width of the original image
  height = num_crops_h * crop_height
  width = num_crops_w * crop_width
  # Initialize the output array with zeros
  output = np.zeros((height, width, N))
  # Loop over the cropped images and copy them to the output array
  for i in range(num_crops_h):
    for j in range(num_crops_w):
      # Get the index of the current cropped image
      k = i * num_crops_w + j
      # Get the start and end coordinates of the output region to paste
      start_y = i * crop_height
      end_y = (i + 1) * crop_height
      start_x = j * crop_width
      end_x = (j + 1) * crop_width
      # Copy the cropped image data to the output array
      output[start_y:end_y, start_x:end_x, :] = cropped_images[k]
  # Return the output array as an integer type (to match the original image type)
  return output.astype(np.uint8)

def predict_litho(img,model):
    crop_sizes = 32*10
    c_img, ri = crop_image_v2(img,(crop_sizes,crop_sizes),255)
    p = []
    for i in range(c_img.shape[0]):
        p.append(model.predict(c_img[i:i+1]/255.0))
    p = np.concatenate(p)

    re_mask = uncrop_image_v2(np.expand_dims(np.argmax(p,-1),3),ri)

    color_map = {}
    colors = random_color(9)
    for i in range(9):
        color_map[i] = colors[i]
    color_map[9] = (255,255,255)

    new_data = np.zeros(re_mask.shape[:2] + (3,), dtype=np.uint8)

    for k, v in color_map.items():
        new_data[re_mask[:, :, 0] == k] = v

    re_mask_l = new_data
    re_img = uncrop_image_v2(c_img,ri), re_mask_l
    
    return re_img, re_mask


def mask_flattened(c_msk):
    # Create an empty array with the same shape as the first three dimensions of c_msk
    result = np.zeros(c_msk.shape[:-1])
    
    # Set the value of result to 1 where any layer of c_msk is active
    result[np.any(c_msk > 0, axis=-1)] = 1
    
    return np.expand_dims(result,len(result.shape))

def predict_colors(re_img,re_msk,N):

    # Load and reshape the image
    image = re_img * mask_flattened(re_msk)
    image[image == 255] = 0
    pixel_vals = image.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)

    # Select only the pixels that are non-zero on all three channels
    non_zero_pixels = non_zero_pixels = pixel_vals[np.nonzero(np.all(pixel_vals != 0, axis=-1) & np.any(pixel_vals != 255, axis=-1))]

    # Apply k-means clustering with k=3 on the non-zero pixels
    kmeans = KMeans(n_clusters=N)
    kmeans.fit(non_zero_pixels)
    centers = kmeans.cluster_centers_
    centers = np.uint8(centers)


    # Print the RGB values of the three most dominant colors
    print("The RGB values of the three most dominant colors are:")
    for i in range(N):
        print(f"Cluster {i}: {centers[i]}")
    
    centers_hex = ['#'+str(rgb_to_hex(tuple(i))) for i in centers]
    
    return centers

def black_to_white(image: np.ndarray) -> np.ndarray:
    # Create a mask for black pixels
    mask = np.all(image == [0, 0, 0], axis=-1)
    
    # Change black pixels to white
    image[mask] = [255, 255, 255]
    
    return image

def get_n_most_common_colors(image_a, n):
    img = Image.fromarray(image_a)
    colors = img.getcolors(img.size[0] * img.size[1])
    sorted_colors = sorted(colors, key=lambda t: t[0], reverse=True)
    return [np.array(color[1]) for color in sorted_colors[:n]]

def predict_mask(img,mask,N):
    centers = get_n_most_common_colors(cv2.bitwise_and(img, img, mask),N)
    threshold = 32
    outputs = []
    for i in range(len(tuple(centers))):
        rgb_color = tuple(centers[i])
        lower = np.array([max(0, c - threshold) for c in rgb_color])
        upper = np.array([min(255, c + threshold) for c in rgb_color])
        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)
        outputs.append(output)
    return outputs, centers


def analyze_mask(mask: np.ndarray, stat: str = 'median', n_v: int = 2) -> np.ndarray:
    """
    Analyze a mask image and return the mean or median position of pixels with value 1 in each row.

    :param mask: A 2D numpy array representing the mask image.
    :param stat: The statistical measure to use ('mean' or 'median').
    :param n_v: The number of neighbors to check in each direction (up and down).
    :return: A tuple containing two 1D numpy arrays representing the filtered X and Y values.
    """
    assert stat in ['mean', 'median'], "Invalid value for 'stat'. Must be 'mean' or 'median'."
    result = []
    for row in mask:
        positions = np.where(row == 1)[0]
        if len(positions) > 0:
            if stat == 'mean':
                result.append(np.mean(positions))
            else:
                result.append(np.median(positions))
        else:
            result.append(mask.shape[1]+1)
    results = zip(range(len(result)),result)
    filtered = []
    for i in results:
        if (i[1] < mask.shape[1]):
            filtered.append(i)
    filtered = np.array(filtered)
    Y = filtered[:,0]
    X = filtered[:,1]
    
    # Additional feature
    X_filtered = []
    Y_filtered = []
    for i in range(len(X)):
        start = max(0, i - n_v)
        end = min(len(X), i + n_v + 1)
        neighbors = X[start:end]
        q1 = np.percentile(neighbors, 25)
        q3 = np.percentile(neighbors, 75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        if lower_bound <= X[i] <= upper_bound:
            X_filtered.append(X[i])
            Y_filtered.append(Y[i])
    
    return (np.array(X_filtered), np.array(Y_filtered))

@st.cache_resource
def load_model():
    host = "https://ai-schlumberger-eag-consulting.p.datascience.cloud.slb-ds.com"
    apiKey = 'CvyAhYL5Y1AO5VLd1aNDUUVSrgXRBFjL'

    client = dataiku.set_remote_dss(host, apiKey)

    model_folder = dataiku.Folder("Module1-2",'MODULE1')
    model_path = model_folder.list_paths_in_partition()

    tmp_file_path = 'model_final_lf.h5'

    with open(tmp_file_path, "wb") as tmp_file:
        with model_folder.get_download_stream(model_path[0]) as model_weight_file:
            tmp_file.write(model_weight_file.read())

    model = tf.keras.models.load_model(tmp_file_path,compile=False)
    os.remove(tmp_file_path)
    return model

#Preprocessing Lib End



col1, col2 = st.sidebar.columns(2)

# Add sliders to control the positions of the horizontal lines in the first column
with col1:
    st.markdown("<div class='column'>", unsafe_allow_html=True)
    st.markdown("<b>Depth-min</b>", unsafe_allow_html=True)
    depth_min= st.number_input("Value max", key="y_max_value")


    # Add sliders to control the positions of the vertical lines in the second column
with col2:
    st.markdown("<div class='column'>", unsafe_allow_html=True)
    st.markdown("<b>Depth-max</b>", unsafe_allow_html=True)
    x_max_value = st.number_input("Value max", key="x_max_value")


# Specify canvas parameters in application
drawing_mode = 'line'

h_line_color_1 = "blue"
h_line_color_2 = "green"
v_line_color_1 = "red"
v_line_color_2 = "black"
bg_color = "#eee"
N=1

st.markdown("<h1 style='text-align: left;'>Upload Lithofacies Image</h1>", unsafe_allow_html=True)
bg_image = st.file_uploader("Upload the Lithofacies Images:", type=["png", "jpg"])

realtime_update = True
accuracy = 1
width = 800
height = 800

canvas_resized = False


if bg_image is not None:
    crop_sizes = 32*10
    image = Image.open(bg_image)
    c_img, ri = crop_image_v2(np.asarray(image),(crop_sizes,crop_sizes),255)
    image = Image.fromarray(uncrop_image_v2(c_img,ri).astype('uint8'), 'RGB')
    width, height = image.size
    max_length = 1500

# Define the custom CSS styles
styles = """
<style>
.column {
    border: 2px solid #ddd;
    border-radius: 10px;
    padding: 10px;
}
</style>


"""

canvas_css = """
.stMarkdown {
    display: grid;
    place-items: center !important;
}
"""
st.markdown(f'<style>{canvas_css}</style>', unsafe_allow_html=True)  # to center title for reference


# Add the custom CSS styles to the page
st.markdown(styles, unsafe_allow_html=True)

# Add a horizontal line to separate the sections
st.sidebar.markdown("<hr/>", unsafe_allow_html=True)

@st.cache_data
def create_images(N):
    return [Image.fromarray(np.full((300, 300, 3), 255, dtype=np.uint8)) for _ in range(N)]

if "images_list" in st.session_state:
    images_list = st.session_state["images_list"]
else:
    images_list = create_images(N)


@st.cache_data
def create_images(N):
    return [Image.fromarray(np.full((300, 300, 3), 255, dtype=np.uint8)) for _ in range(N)]

if "images_output" in st.session_state:
    images_output = st.session_state["images_output"]
else:
    images_output = create_images(N)


@st.cache_data
def create_data():
    return pd.DataFrame([1,2,3])

if "area_data" in st.session_state:
    area_data = st.session_state["area_data"]
else:
    area_data = create_data()


# Calculate the y-coordinates of the horizontal lines and the x-coordinates of the vertical lines based on the slider values

# Create a canvas component
# Create the Predict button outside of any conditional blocks
st.markdown("<h2 style='text-align: left;'></h2>", unsafe_allow_html=True)  
st.markdown("<hr style='border-top: 2px solid ; margin-top: 0;'/>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: left;'>Lithofacies Predictions</h2>", unsafe_allow_html=True)
predict_button = st.button('Digitze Lithofacies')
st.markdown("<hr style='border-top: 2px solid ; margin-top: 0;'/>", unsafe_allow_html=True)




# Define a function that takes the data array, width and height as arguments
def create_image(data, width):
    # Initialize an empty array of shape (height, width, 1) to store the output image
    image = np.zeros((data.shape[0], width, 1))

    # Loop through each row of the data array
    for i in range(data.shape[0]):
        # Get the percentage values of the three classes for the current row
        p1, p2, p3 = data[i]

        # Calculate the number of pixels for each class based on the width and percentage
        if p1+p2+p3 > 50:
            n1 = int(np.floor(p1 * width / 100))
            n2 = int(np.floor(p2 * width / 100))
            n3 = width - n1 - n2

            # Fill the corresponding pixels in the image array with different values for each class
            # For example, you can use 0 for class 1, 127 for class 2 and 255 for class 3
            image[i, :n1, 0] = 30
            image[i, n1:n1+n2, 0] = 127
            image[i, n1+n2:n1+n2+n3, 0] = 180

    # Return the image array
    return image


def percentages(image):
    # Step 1: Calculate the 3 most common pixel values in the image. Ignore pixel of value 9 in this calculation (only consider 0-9). Lets call this the_3
    pixel_values = np.unique(image[image != 9])
    pixel_counts = [len(image[image == i]) for i in pixel_values]
    sorted_pixel_counts = sorted(pixel_counts, reverse=True)
    the_3 = [pixel_values[pixel_counts.index(sorted_pixel_counts[i])] for i in range(3)]
    
    # Step 2: Iterate through every row of the image, calculate the number of the 3 most common pixel, and calculate the percentage of each pixel based on the total number of the_3 in that row. Output the result as a row of 3 column, one for each pixel value in the_3.
    percentages = []
    for row in image:
        row_counts = [len(row[row == i]) for i in the_3]
        total_count = sum(row_counts)
        row_percentages = [count / total_count * 100 if total_count > 0 else 0 for count in row_counts]
        percentages.append(row_percentages)
    
    return percentages

def filter_data(data):
    # Initialize an empty list to store the indices of the rows to keep
    keep = []

    # Loop through each row of the data array
    for i in range(data.shape[0]):
        if data[i,:].sum() > 10:
            keep.append(i)
    return np.take(data, keep, axis=0)

def create_image(data, width):
    # Initialize an empty array of shape (height, width, 1) to store the output image
    image = np.zeros((data.shape[0], width, 1))

    # Loop through each row of the data array
    for i in range(data.shape[0]):
        # Get the percentage values of the three classes for the current row
        p1, p2, p3 = data[i]

        # Calculate the number of pixels for each class based on the width and percentage
        if p1+p2+p3 > 50:
            n1 = int(np.floor(p1 * width / 100))
            n2 = int(np.floor(p2 * width / 100))
            n3 = width - n1 - n2

            # Fill the corresponding pixels in the image array with different values for each class
            # For example, you can use 0 for class 1, 127 for class 2 and 255 for class 3
            image[i, :n1, 0] = 30
            image[i, n1:n1+n2, 0] = 127
            image[i, n1+n2:n1+n2+n3, 0] = 180

    # Return the image array
    return image









if predict_button:
    if "images_list" in st.session_state:
        del st.session_state["images_list"]
    st.markdown("<h2 style='text-align: center;'>Output Result 📝</h2>", unsafe_allow_html=True)
    st.success('Connecting to Dataiku DSS Platform')
    model = load_model()
    st.success('Model Successfully Loaded From Delfi')
    image = Image.open(bg_image)
    re_img,re_mask, re_mask_l = predict_litho(np.asarray(image),model)
    st.success('Prediction Done ! Showing Prediction')
    image = re_img
    images_list = [re_mask_l]
    st.session_state["images_list"] = images_list
    area_data = filter_data(np.array(percentages(re_mask)))
    images_output = create_image(area_data, 1000)


st.markdown(
    """
    <style>
        h2 {
            font-weight: 200;
        }
    </style>
    """,
    unsafe_allow_html=True
)

col3, col4, col5 = st.columns((1,1,1))
if bg_image is not None:
    with col3:
        st.image(image,width=300)
    with col4:
        st.image(images_list[0],width=300)
    with col5:
        st.image(images_output[0],width=images_output[1]/image.shape[1]*300)
else:
    with col4:
        st.image(images_list[0],width=300)
    with col3:
        st.image(images_list[0],width=300)
    with col5:
        st.image(images_output[0],width=images_output.shape[1]/images_list[0].shape[1]*300)
# Define the predict_button variable before it is used

def get_table_download_link(df):
        """Generates a link allowing the data in a given panda dataframe to be downloaded
        in:  dataframe
        out: href string
        """
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
        return href


def calculate_and_download_values():
    
    df = pd.DataFrame(area_data,columns=[0,1,2])
    df['Depth'] = np.linspace(40,100,df.shape[0])
    # Download the DataFrame as an Excel file
    st.session_state['df'] = df


st.markdown("<h2 style='text-align: left;'>Calculate and Download Values</h2>", unsafe_allow_html=True)
calculate_button = st.button('Generate Download Links', on_click=calculate_and_download_values)

if 'df' in st.session_state:
        st.markdown(get_table_download_link(st.session_state['df']), unsafe_allow_html=True)
