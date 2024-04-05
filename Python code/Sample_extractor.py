import os
import random
import shutil

os.chdir("C:\\Users\\MusondaSinkala\\Documents\\Education\\MS Data Science\\Spring 2024\\Machine Learning\\Project\\")
os.getcwd()


# Path to the root directory where the dataset is located
root_dir = 'Data\\'

# List all patient IDs
patient_ids = [patient_id for patient_id in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, patient_id))]

# Select 100 random patient IDs
random_patient_ids = random.sample(patient_ids, 100)

# Create a new directory for the random sample
sample_dir = os.path.join(root_dir, 'Random Sample')
os.makedirs(sample_dir, exist_ok=True)

# Copy the selected patient folders to the sample directory
for patient_id in random_patient_ids:
    src_path = os.path.join(root_dir, patient_id)
    dst_path = os.path.join(sample_dir, patient_id)
    shutil.copytree(src_path, dst_path)

print("Random sample of 100 patients copied to:", sample_dir)
