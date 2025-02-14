import pandas as pd
import numpy as np
import os
from urllib import request
import shutil
import math
from concurrent.futures import ThreadPoolExecutor, as_completed


class CsvToYolo:

    def __init__(self, input_csv_file_path, output_folder, validation_split=0.1, download_images=False, google_api_key_filepath=None, image_size=(640, 640), max_workers=8):
        self.input_csv_file_path = input_csv_file_path
        self.output_folder = output_folder
        self.validation_split = validation_split
        self.download_images = download_images
        self.image_width, self.image_height = image_size
        self.yolo_data = []
        self.num_subsets = 5
        self.max_workers = max_workers  # Number of threads for parallel execution

        # Folder paths
        self.image_folder_root = os.path.join(self.output_folder, 'images')
        self.label_folder_root = os.path.join(self.output_folder, 'labels')

        # Create required directories only when needed
        if self.download_images:
            for i in range(1, self.num_subsets + 1):
                os.makedirs(os.path.join(self.image_folder_root, f'image_subset_{i}'), exist_ok=True)
            os.makedirs(os.path.join(self.image_folder_root, 'validation'), exist_ok=True)

            with open(google_api_key_filepath, 'r') as f:
                self.api_key = f.read().strip()

        for i in range(1, self.num_subsets + 1):
            os.makedirs(os.path.join(self.label_folder_root, f'label_subset_{i}'), exist_ok=True)
        os.makedirs(os.path.join(self.label_folder_root, 'validation'), exist_ok=True)

    def convert_to_yolo_format(self, min_x, min_y, max_x, max_y):
        center_x = (min_x + max_x) / 2.0 / self.image_width
        center_y = (min_y + max_y) / 2.0 / self.image_height
        width = (max_x - min_x) / self.image_width
        height = (max_y - min_y) / self.image_height
        return center_x, center_y, width, height

    def download_image(self, url, image_path):
        try:
            request.urlretrieve(url, image_path)
            return f"Downloaded {image_path}"
        except Exception as e:
            return f"Failed to download {image_path}: {str(e)}"

    def process_helipads(self, df_helipad, helipad_number, label_folder, image_folder=None):
        yolo_labels = []
        url = df_helipad.iloc[0]['url'].replace('APIKEY', self.api_key)

        for _, row in df_helipad.iterrows():
            min_x, min_y, max_x, max_y = row[['minX', 'minY', 'maxX', 'maxY']]
            x_center, y_center, width, height = self.convert_to_yolo_format(min_x, min_y, max_x, max_y)
            yolo_labels.append(f"0 {x_center} {y_center} {width} {height}")
            self.yolo_data.append([helipad_number, x_center, y_center, width, height])

        # Save YOLO label
        label_filename = f"Helipad_{helipad_number:05d}.txt"
        label_path = os.path.join(label_folder, label_filename)
        with open(label_path, 'w') as f:
            for label in yolo_labels:
                f.write(label + '\n')

        # Parallel download image if needed
        if self.download_images and image_folder:
            image_filename = f"Helipad_{helipad_number:05d}.png"
            image_path = os.path.join(image_folder, image_filename)
            return self.download_image(url, image_path)

        return f"Processed labels for Helipad_{helipad_number}"

    def run(self):
        df = pd.read_csv(self.input_csv_file_path)

        helipad_numbers = df.Helipad_number.unique().tolist()
        np.random.shuffle(helipad_numbers)

        total_helipads = len(helipad_numbers)
        validation_size = int(total_helipads * self.validation_split)

        # Split into training and validation
        validation_helipads = helipad_numbers[:validation_size]
        train_helipads = helipad_numbers[validation_size:]

        # 80% of the training data to be shared across all subsets
        common_helipads_size = int(0.8 * len(train_helipads))
        common_helipads = train_helipads[:common_helipads_size]

        # The remaining 20% will be different for each subset
        remaining_helipads = train_helipads[common_helipads_size:]
        unique_size_per_subset = len(remaining_helipads) // self.num_subsets

        subsets = []
        for i in range(self.num_subsets):
            start_idx = i * unique_size_per_subset
            end_idx = (i + 1) * unique_size_per_subset
            unique_helipads = remaining_helipads[start_idx:end_idx]

            # Combine the common 80% with the unique 20% for each subset
            subset_helipads = common_helipads + unique_helipads
            subsets.append(subset_helipads)

        # Process validation set
        validation_folder = os.path.join(self.label_folder_root, 'validation')
        image_validation_folder = os.path.join(self.image_folder_root, 'validation') if self.download_images else None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            # Process validation data
            for helipad_number in validation_helipads:
                df_helipad = df[df['Helipad_number'] == helipad_number]
                futures.append(executor.submit(self.process_helipads, df_helipad, helipad_number, validation_folder, image_validation_folder))

            # Process training subsets
            for idx, subset in enumerate(subsets):
                label_folder = os.path.join(self.label_folder_root, f'label_subset_{idx + 1}')
                image_folder = os.path.join(self.image_folder_root, f'image_subset_{idx + 1}') if self.download_images else None

                for helipad_number in subset:
                    df_helipad = df[df['Helipad_number'] == helipad_number]
                    futures.append(executor.submit(self.process_helipads, df_helipad, helipad_number, label_folder, image_folder))

            for future in as_completed(futures):
                print(future.result())

        # Save YOLO data to CSV
        yolo_df = pd.DataFrame(self.yolo_data, columns=['Helipad_number', 'x_center', 'y_center', 'width', 'height'])
        csv_output_path = os.path.join(self.output_folder, "helipad_yolo_data.csv")
        yolo_df.to_csv(csv_output_path, index=False)
        print(f'YOLO data saved to {csv_output_path}')


if __name__ == "__main__":
    input_csv_file_path = "./data/Helipad_DataBase_annotated.csv"
    output_folder = "./helipad_images1"
    download_images = True
    google_api_key_filepath = "./api.csv"

    csv_to_yolo = CsvToYolo(input_csv_file_path=input_csv_file_path,
                           output_folder=output_folder,
                           download_images=download_images,
                           google_api_key_filepath=google_api_key_filepath)

    csv_to_yolo.run()
