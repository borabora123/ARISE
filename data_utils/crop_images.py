import os
import argparse
import pandas as pd
import cv2
import json


# Function to crop and save images
def crop_and_save_images(merged_df, image_dir, output_dir, normalized=False):
    """
    Crop images based on bounding boxes and save them with the appropriate label.

    Args:
        merged_df (pd.DataFrame): Merged DataFrame containing bbox and label information.
        image_dir (str): Path to the directory containing input images.
        output_dir (str): Path to save the cropped images.
        normalized (bool): Whether the bounding box coordinates are normalized.
    """
                  
    os.makedirs(output_dir, exist_ok=True)

    for _, row in merged_df.iterrows():
        image_name = str(row["patient_id"]) + ".jpeg"  # Assuming the image name is in a column called "image_name"
        label = f"{int(row['erosion_score'])}_{int(row['jsn_score'])}"
        x_center, y_center, width, height = row["xcenter"], row["ycenter"], row["dx"], row["dy"]

        # Load the image
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Get image dimensions
        img_height, img_width = image.shape[:2]

        # Convert normalized coordinates to pixel coordinates if necessary
        if normalized:
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

        # Calculate bounding box coordinates
        x1 = max(int(x_center - width / 2), 0)
        y1 = max(int(y_center - height / 2), 0)
        x2 = min(int(x_center + width / 2), img_width)
        y2 = min(int(y_center + height / 2), img_height)

        # Crop the image
        cropped_image = image[y1:y2, x1:x2]

        # Save the cropped image
        output_name = f"{os.path.splitext(image_name)[0]}_{row['joint_id']}_{int(row['erosion_score'])}_{int(row['jsn_score'])}.jpg"
        output_path = os.path.join(output_dir, output_name)
        try:
            print(output_path)
            cv2.imwrite(output_path, cropped_image)
        except:
            print(cropped_image.shape)
            print(row)
            return 0


def main():
    # Load the CSV files
    label_df = pd.read_csv('C:\\Users\\User\\PycharmProjects\\ARISE\\dataset\\merges_box_scores.csv')

    # Crop and save images
    crop_and_save_images(label_df, 'C:\\Users\\User\\PycharmProjects\\ARISE\\dataset\\jpeg', 'C:\\Users\\User\\PycharmProjects\\ARISE\\data\\croped_images', normalized=False)


if __name__ == "__main__":
    main()