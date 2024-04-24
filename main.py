import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def simplify_colors(image_path, k=8):
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Implementing k-means clustering to simplify color space
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers to integers and assign colors back to the original pixels
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    # Reshape data into the original image dimensions
    segmented_image = segmented_data.reshape(img.shape)
    return segmented_image, labels.reshape(img.shape[:-1]), centers

def outline_and_label(segmented_image, labels, centers):
    # Create a blank white canvas
    white_canvas = np.ones(segmented_image.shape, dtype=np.uint8) * 255

    # Putting labels on the contours
    for label in np.unique(labels):
        # Create a mask for each segment
        mask = np.uint8(labels == label)
        # Find contours for the current segment
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Simplify contours
        approx_contours = [cv2.approxPolyDP(contour, 0.001*cv2.arcLength(contour, True), True) for contour in contours]
        
        # Draw simplified contours
        cv2.drawContours(white_canvas, approx_contours, -1, (0, 0, 0), 1)

        # Compute the center of the largest contour
        if approx_contours:
            largest_contour = max(approx_contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # Put a label number in the center
                cv2.putText(white_canvas, str(label + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return white_canvas

def plot_color_map(centers):
    # Create a color map using the segmented colors
    cmap = ListedColormap([color / 255 for color in centers])

    # Plot the color map in full screen
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    bounds = np.arange(0, len(centers) + 1)
    norm = plt.Normalize(bounds.min(), bounds.max())
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      ax=ax, orientation='horizontal', ticks=bounds[:-1] + 0.5)
    cb.set_label('Segment')
    cb.ax.set_xticklabels([str(i) for i in range(1, len(centers) + 1)])
    cb.ax.tick_params(axis='x', direction='out', length=3, width=1)
    plt.show()

if __name__ == "__main__":
    image_path = 'imgs\catmill.jpg'
    segmented_image, labels, centers = simplify_colors(image_path, k=40)  # Adjust k as needed
    outlined_image = outline_and_label(segmented_image, labels, centers)

    # Display the segmented image
    plt.imshow(segmented_image)  
    plt.axis('off')
    plt.show()

    # Display the outlined image
    plt.imshow(outlined_image, cmap='gray')  # Display in grayscale
    plt.axis('off')
    plt.show()

    # Plot the color map
    plot_color_map(centers)
    #plt.show()
