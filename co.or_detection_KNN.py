# Required installations: pandas, opencv-python, scikit-learn
import cv2
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Paths to the image and the CSV file containing color data
img_path = 'pic3.jpg'
csv_path = 'colors.csv'

# Reading the CSV file containing the colors
# The CSV should have columns: color, color_name, hex, R, G, B
index = ['color', 'color_name', 'hex', 'R', 'G', 'B']
df = pd.read_csv(csv_path, names=index, header=None)

# Preparing data for KNN
# Extracting RGB values and the corresponding color names
X = df[['R', 'G', 'B']].values
y = df['color_name'].values

# Creating the KNN model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

# Reading and resizing the image
img = cv2.imread(img_path)
img = cv2.resize(img, (800, 600))

# Global variables
clicked = False
r = g = b = xpos = ypos = 0

# Function to get the color name using KNN
def get_color_name(R, G, B):
    global knn
    color = knn.predict([[R, G, B]])
    return color[0]

# Mouse callback function
def draw_function(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, xpos, ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b, g, r = img[y, x]
        b, g, r = int(b), int(g), int(r)

# Creating window and setting the mouse callback function
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_function)

while True:
    cv2.imshow('image', img)
    if clicked:
        # Drawing a rectangle and displaying the color name and RGB values
        cv2.rectangle(img, (20, 20), (600, 60), (b, g, r), -1)
        text = get_color_name(r, g, b) + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)
        cv2.putText(img, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if r + g + b >= 600:
            cv2.putText(img, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    if cv2.waitKey(20) & 0xFF == 27:  # Break the loop when 'Esc' is pressed
        break

cv2.destroyAllWindows()