import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import random

class LabelGenerator:
    def __init__(self, class_csv_path, label_by='class_name'):
        """
        Initialize the LabelGenerator with class mapping.

        Parameters:
        - class_csv_path (str): Path to the class.csv file.
        - label_by (str): Column name to use for labeling ('class_name' or 'name').
        """
        # Load class mapping from CSV
        self.class_df = pd.read_csv(class_csv_path)
        if label_by not in self.class_df.columns:
            raise ValueError(f"'{label_by}' column not found in class CSV.")
        self.label_by = label_by
        
        # Create a mapping from label_by to class_number
        self.class_map = self.class_df.set_index(label_by)['class_number'].to_dict()

        self.image_paths = []
        self.label_paths = []
        self.output_paths = []

    def generate_label_image(self, label_csv_path, image_path, output_path):
        """
        Generate a labeled image based on the input CSV.

        Parameters:
        - label_csv_path (str): Path to the label.csv file containing pixel information.
        - image_path (str): Path to the input image used to determine dimensions.
        - output_path (str): Path to save the output TIF image.
        """
        # Load the input image to determine its dimensions
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load without color conversion
        if image is None:
            raise ValueError(f"Cannot read image at {image_path}")

        if len(image.shape) == 3:  # Color image
            image_height, image_width, _ = image.shape
        elif len(image.shape) == 2:  # Grayscale image
            image_height, image_width = image.shape
        else:
            raise ValueError("Unexpected image shape. Expected 2D or 3D array.")

        # Load label data from CSV
        label_df = pd.read_csv(label_csv_path)

        # Create an empty image (16-bit grayscale)
        label_image = np.zeros((image_height, image_width), dtype=np.uint16)

        # Populate the label image
        for _, row in label_df.iterrows():
            x, y = int(row['x']), int(row['y'])
            label_value = row[self.label_by]

            # Get the class number and compute pixel value
            class_number = self.class_map.get(label_value, 0)  # Default to 0 if not found
            pixel_value = class_number + 1  # Add 1 to class_number

            # Set the pixel value at (x, y)
            if 0 <= y < image_height and 0 <= x < image_width:
                label_image[y, x] = pixel_value
            else:
                print(f"Skipping out-of-bounds pixel at ({x}, {y})")

        # Save the output image as a TIF file (16-bit grayscale)
        cv2.imwrite(output_path, label_image)
        print(f"Label image saved to {output_path}")

    def create_paths(self, images_dir, labels_dir, target_output):
        # สร้างโฟลเดอร์ target_output หากยังไม่มี
        if not os.path.exists(target_output):
            os.makedirs(target_output, exist_ok=True)

        self.image_paths = []
        self.label_paths = []
        self.output_paths = []

        for subdir, _, files in os.walk(images_dir):
            for file in files:
                # ตรวจสอบว่าเป็นไฟล์ภาพ
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                    image_path = os.path.join(subdir, file)
                    
                    # เปลี่ยนนามสกุลไฟล์เอาต์พุตเป็น .tif
                    output_file_name = os.path.splitext(file)[0] + ".tif"
                    output_path = os.path.join(target_output, output_file_name)

                    # สร้างพาธของ label_path
                    base_name = os.path.splitext(file)[0]  # เอาเฉพาะชื่อไฟล์ (ไม่มีนามสกุล)
                    label_path = os.path.join(labels_dir, f"{base_name}.csv")

                    # ตรวจสอบว่าไฟล์ label มีอยู่จริง
                    if os.path.exists(label_path):
                        if image_path not in self.image_paths:  # ป้องกันการเพิ่มซ้ำ
                            self.image_paths.append(image_path)
                            self.label_paths.append(label_path)
                            self.output_paths.append(output_path)
                    else:
                        print(f"Warning: Label file not found for {file}")

    def generate_label_image_all(self, do_expand=False, dilation_radius=5, color_tolerance=50):
        if not (self.image_paths and self.label_paths):
            print("No paths available. Run `create_paths` first to populate paths.")
            return

        for i in range(len(self.image_paths)):
            if not do_expand:
                self.generate_label_image(
                    image_path=self.image_paths[i],
                    label_csv_path=self.label_paths[i],
                    output_path=self.output_paths[i]
                )
            else:
                self.expand_labels_by_color(
                    image_path=self.image_paths[i],
                    label_path=self.label_paths[i],
                    output_path=self.output_paths[i],
                    dilation_radius=5,
                    color_tolerance=color_tolerance
                )
        print("Generate image labels completed")

    def expand_labels_by_color(self, image_path, label_path, output_path, dilation_radius=5, color_tolerance=50):
        """
        Expand labels in an image by similar colors and dilate the labeled regions.

        Parameters:
        - image_path (str): Path to the input image.
        - label_path (str): Path to the label CSV file.
        - output_path (str): Path to save the output expanded label image.
        - dilation_radius (int): Radius for morphological dilation.
        - color_tolerance (int): Tolerance for color similarity (0-255).
        """
        # Load image and labels
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load color image
        label_df = pd.read_csv(label_path)

        if image is None:
            raise ValueError(f"Cannot read image at {image_path}")

        # Get dimensions of the image
        image_height, image_width, _ = image.shape

        # Create an empty label image
        label_image = np.zeros((image_height, image_width), dtype=np.uint16)

        # Populate initial labels
        for _, row in label_df.iterrows():
            x, y = int(row['x']), int(row['y'])
            label_value = row[self.label_by]
            class_number = self.class_map.get(label_value, 0)  # Default to 0 if not found
            pixel_value = class_number + 1

            if 0 <= y < image_height and 0 <= x < image_width:
                label_image[y, x] = pixel_value

        # Expand labels by color similarity
        expanded_label_image = np.copy(label_image)
        for y in range(image_height):
            for x in range(image_width):
                if label_image[y, x] > 0:  # If the pixel is labeled
                    pixel_color = image[y, x]
                    for i in range(y - dilation_radius, y + dilation_radius + 1):
                        for j in range(x - dilation_radius, x + dilation_radius + 1):
                            # Ensure neighbor coordinates are within image bounds
                            if 0 <= i < image_height and 0 <= j < image_width:
                                neighbor_color = image[i, j]
                                pixel_color = pixel_color.astype(int)  # Convert to int
                                neighbor_color = neighbor_color.astype(int)
                                color_distance = np.linalg.norm(pixel_color - neighbor_color)
                                if color_distance <= color_tolerance and expanded_label_image[i, j] == 0:
                                    expanded_label_image[i, j] = label_image[y, x]

        # Apply morphological dilation using OpenCV
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_radius + 1, 2 * dilation_radius + 1))
        dilated_label_image = cv2.dilate(expanded_label_image, kernel)

        # Save the output image as a TIF file
        cv2.imwrite(output_path, dilated_label_image.astype(np.uint16))
        print(f"Expanded label image saved to {output_path}")

    def expand_labels_by_color2(self, image_path, label_path, output_path, dilation_radius=5, color_tolerance=1, roi=None): # not good
        """
        Expand labels in an image by similar colors and dilate the labeled regions.

        Parameters:
        - image_path (str): Path to the input image.
        - label_path (str): Path to the label CSV file.
        - output_path (str): Path to save the output expanded label image.
        - dilation_radius (int): Radius for morphological dilation.
        - color_tolerance (int): Tolerance for color similarity (0-255).
        - roi (tuple): Region of Interest in the format (x_min, y_min, x_max, y_max). Default is None (full image).
        """
        # Load image and labels
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load color image
        label_df = pd.read_csv(label_path)

        if image is None:
            raise ValueError(f"Cannot read image at {image_path}")

        # Get dimensions of the image
        image_height, image_width, _ = image.shape

        # Set ROI boundaries
        if roi is not None:
            x_min, y_min, x_max, y_max = roi
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(image_width, x_max), min(image_height, y_max)
        else:
            x_min, y_min, x_max, y_max = 0, 0, image_width, image_height

        # Create an empty label image
        label_image = np.zeros((image_height, image_width), dtype=np.uint16)

        # Populate initial labels within ROI
        for _, row in label_df.iterrows():
            x, y = int(row['x']), int(row['y'])
            if x_min <= x < x_max and y_min <= y < y_max:
                label_value = row[self.label_by]
                class_number = self.class_map.get(label_value, 0)  # Default to 0 if not found
                pixel_value = class_number + 1
                label_image[y, x] = pixel_value

        # Expand labels by color similarity within ROI
        expanded_label_image = np.copy(label_image)
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if label_image[y, x] > 0:  # If the pixel is labeled
                    pixel_color = image[y, x].astype(int)  # Convert to int
                    lower_bound = np.maximum(pixel_color - color_tolerance, 0)
                    upper_bound = np.minimum(pixel_color + color_tolerance, 255)

                    mask = np.zeros((image_height + 2, image_width + 2), dtype=np.uint8)  # Mask for floodFill

                    cv2.floodFill(
                        image,
                        mask,
                        (x, y),
                        pixel_value,
                        loDiff=lower_bound.tolist(),
                        upDiff=upper_bound.tolist(),
                        flags=cv2.FLOODFILL_FIXED_RANGE
                    )

                    # Update expanded label image with mask
                    label_mask = mask[1:-1, 1:-1]  # Remove padding
                    expanded_label_image[label_mask > 0] = label_image[y, x]

        # Apply morphological dilation within ROI
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_radius + 1, 2 * dilation_radius + 1))
        expanded_label_image[y_min:y_max, x_min:x_max] = cv2.dilate(
            expanded_label_image[y_min:y_max, x_min:x_max], kernel, iterations=1
        )

        # Save the output image as a TIF file
        cv2.imwrite(output_path, expanded_label_image.astype(np.uint16))
        print(f"Expanded label image saved to {output_path}")

    def resize_image_and_labels(self, image_path, label_csv_path, output_image_path, output_label_path, width, height):
        """
        Resize the image to the specified width and height, and adjust label coordinates accordingly.

        Parameters:
        - image_path (str): Path to the original image file.
        - label_csv_path (str): Path to the original label CSV file.
        - output_image_path (str): Path to save the resized image.
        - output_label_path (str): Path to save the resized label CSV.
        - width (int): Desired width of the resized image.
        - height (int): Desired height of the resized image.
        """
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read image at {image_path}")

        # Get original dimensions
        orig_height, orig_width = image.shape[:2]

        # Resize the image
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_image_path, resized_image)
        print(f"Resized image saved to {output_image_path}")

        # Load the label CSV
        label_df = pd.read_csv(label_csv_path)

        # Adjust label coordinates based on the new image dimensions
        label_df['x'] = (label_df['x'] * width / orig_width).astype(int)
        label_df['y'] = (label_df['y'] * height / orig_height).astype(int)

        # Save the updated label CSV
        label_df.to_csv(output_label_path, index=False)
        print(f"Resized label CSV saved to {output_label_path}")

    def resize_image_and_labels_all(self, output_images_dir, output_labels_dir, width, height):
        """
        Resize all images and labels, saving them to the specified directories.

        Parameters:
        - output_images_dir (str): Directory to save resized images.
        - output_labels_dir (str): Directory to save resized label CSV files.
        - width (int): Desired width of the resized images.
        - height (int): Desired height of the resized images.
        """
        # ตรวจสอบและสร้างโฟลเดอร์ output ถ้ายังไม่มี
        if not os.path.exists(output_images_dir):
            os.makedirs(output_images_dir)
            print(f"Created directory: {output_images_dir}")
        if not os.path.exists(output_labels_dir):
            os.makedirs(output_labels_dir)
            print(f"Created directory: {output_labels_dir}")

        output_image_paths = []
        output_label_paths = []

        for i in range(len(self.image_paths)):
            # ดึงชื่อไฟล์ภาพและ label
            image_name = os.path.basename(self.image_paths[i])  # ชื่อไฟล์ภาพ
            label_name = os.path.basename(self.label_paths[i])  # ชื่อไฟล์ label (CSV)

            # สร้างพาธสำหรับไฟล์ output
            output_image_path = os.path.join(output_images_dir, image_name)
            output_label_path = os.path.join(output_labels_dir, label_name)

            # เรียกฟังก์ชัน resize สำหรับแต่ละภาพและ label
            self.resize_image_and_labels(
                self.image_paths[i],
                self.label_paths[i],
                output_image_path,
                output_label_path,
                width,
                height
            )

    def save_image_with_overlay(self, original_image_path, label_image_path, image_with_overlay_path, overlay_path):
        """
        Plot an original image with an overlay of label image.

        Parameters:
        - original_image_path (str): Path to the original image file.
        - label_image_path (str): Path to the label image file (TIF, 16-bit grayscale).
        """
        # Load original image
        original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)  # Load color image (BGR format)
        if original_image is None:
            raise ValueError(f"Cannot read image at {original_image_path}")

        # Convert BGR to RGB
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Load label image
        label_image = cv2.imread(label_image_path, cv2.IMREAD_UNCHANGED)  # Load TIF (16-bit grayscale)
        if label_image is None:
            raise ValueError(f"Cannot read label image at {label_image_path}")
        
        # Define colors for 22 labels
        color_b = [0,   0,   0, 255,   0, 255, 255,   255, 128, 64, 128, 192, 192, 64, 192,   0, 128, 128, 64, 64, 128, 192]
        color_g = [0,   0, 255,   0, 255,   0, 255,   255, 128, 64, 192, 192, 64,  0,  64, 192,  64, 192,  0, 64,  64,   0]
        color_r = [0, 255,   0,   0, 255, 255,   0,   255, 128, 64, 128,  64, 64, 64,   0,   0, 192,  64, 64,  0, 128, 192]

        # Normalize label_image to fit the range [0-21]
        label_image = np.clip(label_image, 0, 21).astype(np.uint8)

        # Create overlay color image
        overlay = np.zeros((label_image.shape[0], label_image.shape[1], 3), dtype=np.uint8)
        overlay[:, :, 0] = np.take(color_r, label_image)  # Red channel
        overlay[:, :, 1] = np.take(color_g, label_image)  # Green channel
        overlay[:, :, 2] = np.take(color_b, label_image)  # Blue channel

        # Create alpha mask where label_image > 0
        alpha = (label_image > 0).astype(np.float32) * 0.6  # 60% transparency

        # Normalize original_image if needed
        if original_image.dtype != np.uint8:
            original_image = ((original_image - original_image.min()) / (original_image.max() - original_image.min()) * 255).astype(np.uint8)

        # Ensure original image is RGB
        if len(original_image.shape) == 2:  # Grayscale
            original_image = np.stack([original_image] * 3, axis=-1)

        # Blend original image and overlay
        blended = (original_image * (1 - alpha)[:, :, None] + overlay * alpha[:, :, None]).astype(np.uint8)

        cv2.imwrite(overlay_path, overlay)
        cv2.imwrite(image_with_overlay_path, blended)

    def save(self, output_images_dir, output_labels_dir):
        if not os.path.exists(output_images_dir):
            os.makedirs(output_images_dir)
            print(f"Created directory: {output_images_dir}")
        if not os.path.exists(output_labels_dir):
            os.makedirs(output_labels_dir)
            print(f"Created directory: {output_labels_dir}")

        for i in range(len(self.image_paths)):
            image_name = os.path.basename(self.image_paths[i])  # ชื่อไฟล์ภาพ

            output_image_path = os.path.join(output_images_dir, image_name)
            output_overlay_path = os.path.join(output_labels_dir, image_name)
            self.save_image_with_overlay(self.image_paths[i], self.output_paths[i], output_image_path, output_overlay_path)

    def apply_mask_points_all(self, mask_point_dir):
        """
        Generate binary mask images from label images and save them to the specified directory.

        Parameters:
        - mask_point_dir (str): Directory to save the generated mask images.
        """
        # ตรวจสอบหรือสร้างโฟลเดอร์เป้าหมาย
        if not os.path.exists(mask_point_dir):
            os.makedirs(mask_point_dir)
            print(f"Created directory: {mask_point_dir}")

        # ประมวลผลแต่ละ output label image
        for i in range(len(self.output_paths)):
            image_name = os.path.basename(self.output_paths[i])  # ดึงชื่อไฟล์จากพาธ
            output_mask_path = os.path.join(mask_point_dir, image_name)

            # โหลด label image
            label_image = cv2.imread(self.output_paths[i], cv2.IMREAD_UNCHANGED)  # โหลด TIF (16-bit grayscale)
            if label_image is None:
                print(f"Warning: Could not load label image at {self.output_paths[i]}")
                continue

            # สร้าง mask โดยกำหนดค่าพิกเซล > 0 ให้เป็น 1
            mask_image = (label_image > 0).astype(np.uint16)  # แปลงเป็น uint16

            # บันทึก mask image เป็น TIF (16-bit grayscale)
            cv2.imwrite(output_mask_path, mask_image)
            print(f"Saved mask image: {output_mask_path}")

    def split_data_to_txt(self, image_dir, mask_dir, segment_dir, output_dir, train_ratio=0.8):
        """
        Split the dataset into training and validation sets and save the file paths to .txt files.

        Parameters:
        - image_dir (str): Directory containing images.
        - mask_dir (str): Directory containing masks.
        - segment_dir (str): Directory containing segmentation files.
        - output_dir (str): Directory to save the split .txt files.
        - train_ratio (float): Proportion of data to use for training (0.0 to 1.0).
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        # รวมข้อมูลทั้งหมด
        data = list(zip(self.image_paths, self.output_paths))
        random.shuffle(data)  # สุ่มข้อมูล

        # คำนวณจำนวนไฟล์สำหรับ train set
        train_size = int(len(data) * train_ratio)

        train_data = data[:train_size]
        val_data = data[train_size:]

        def write_set_to_file(file_path, dataset, image_dir, mask_dir, segment_dir):
            with open(file_path, 'w') as f:
                f.write(f"image_dir={image_dir}\n")
                f.write(f"mask_dir={mask_dir}\n")
                f.write(f"segment_dir={segment_dir}\n")
                f.write("image_filename, mask_filename, segmentation_filename, coral_count\n")

                for image_path, segment_path in dataset:
                    image_filename = os.path.basename(image_path)
                    mask_filename = os.path.basename(segment_path)  # ใช้ชื่อเดียวกับ segment_path
                    segment_filename = os.path.basename(segment_path)

                    # สร้าง path สำหรับ segmentation file
                    segment_full_path = os.path.join(segment_dir, segment_filename)

                    # โหลด label image
                    label_image = cv2.imread(segment_full_path, cv2.IMREAD_UNCHANGED)
                    if label_image is None:
                        print(f"Warning: Unable to read {segment_full_path}. Skipping...")
                        coral_count = 0
                    else:
                        coral_count = (label_image == 1).sum()  # นับ pixel ที่มีค่าเป็น 1

                    # เขียนข้อมูลลงไฟล์
                    f.write(f"{image_filename}, {mask_filename}, {segment_filename}, {coral_count}\n")
            print(f"Set saved to {file_path}")

        # เขียน train_set.txt
        train_file_path = os.path.join(output_dir, 'train_set.txt')
        write_set_to_file(train_file_path, train_data, image_dir, mask_dir, segment_dir)

        # เขียน val_set.txt
        val_file_path = os.path.join(output_dir, 'val_set.txt')
        write_set_to_file(val_file_path, val_data, image_dir, mask_dir, segment_dir)


def plot_image_with_overlay(original_image_path, label_image_path, save_overlay=False):
    """
    Plot an original image with an overlay of label image.

    Parameters:
    - original_image_path (str): Path to the original image file.
    - label_image_path (str): Path to the label image file (TIF, 16-bit grayscale).
    """
    # Load original image
    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)  # Load color image (BGR format)
    if original_image is None:
        raise ValueError(f"Cannot read image at {original_image_path}")

    # Convert BGR to RGB
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Load label image
    label_image = cv2.imread(label_image_path, cv2.IMREAD_UNCHANGED)  # Load TIF (16-bit grayscale)
    if label_image is None:
        raise ValueError(f"Cannot read label image at {label_image_path}")
    
    # Define colors for 22 labels
    color_b = [0,   0,   0, 255,   0, 255, 255,   255, 128, 64, 128, 192, 192, 64, 192,   0, 128, 128, 64, 64, 128, 192]
    color_g = [0,   0, 255,   0, 255,   0, 255,   255, 128, 64, 192, 192, 64,  0,  64, 192,  64, 192,  0, 64,  64,   0]
    color_r = [0, 255,   0,   0, 255, 255,   0,   255, 128, 64, 128,  64, 64, 64,   0,   0, 192,  64, 64,  0, 128, 192]

    # Normalize label_image to fit the range [0-21]
    label_image = np.clip(label_image, 0, 21).astype(np.uint8)

    # Create overlay color image
    overlay = np.zeros((label_image.shape[0], label_image.shape[1], 3), dtype=np.uint8)
    overlay[:, :, 0] = np.take(color_r, label_image)  # Red channel
    overlay[:, :, 1] = np.take(color_g, label_image)  # Green channel
    overlay[:, :, 2] = np.take(color_b, label_image)  # Blue channel

    # Create alpha mask where label_image > 0
    alpha = (label_image > 0).astype(np.float32) * 0.6  # 60% transparency

    # Normalize original_image if needed
    if original_image.dtype != np.uint8:
        original_image = ((original_image - original_image.min()) / (original_image.max() - original_image.min()) * 255).astype(np.uint8)

    # Ensure original image is RGB
    if len(original_image.shape) == 2:  # Grayscale
        original_image = np.stack([original_image] * 3, axis=-1)

    # Blend original image and overlay
    blended = (original_image * (1 - alpha)[:, :, None] + overlay * alpha[:, :, None]).astype(np.uint8)

    # Plot the result
    plt.figure(figsize=(10, 10))
    plt.imshow(blended)
    plt.axis('off')
    plt.title("Image with Label Overlay")
    plt.show()

    if(save_overlay):
        cv2.imwrite("overlay.jpg",overlay)