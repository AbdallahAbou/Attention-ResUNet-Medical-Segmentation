"""
Author:    FName LName
Created:   DD.MM.YYYY

Description

Example call
main.py /job /download /output /datasets

python main.py --ScratchDir ./job --DownloadDir ./download --OutputDir ./output --DataDir ./dataset

"""
import os
import argparse
from src.data.download_and_prepare_data import download_and_prepare_data
from src.data.process_data import process_all_data
from src.data.check_flag import check_flag_status, set_flag_status
from src.models.train_model import train_model


# Create parser
parser = argparse.ArgumentParser(
    prog="main.py", description="Execute during SLURM job.", add_help=True
)

# Create parser arguments

parser.add_argument(
    "--ScratchDir",
    type=str,
    nargs="?",
    default="./",
    help="Sets job directory. Provided by slurm when using BMI cluster. Leave empty for testing. Defaults to ./",
)
parser.add_argument(
    "--DownloadDir",
    type=str,
    nargs="?",
    default="./",
    help="Sets download directory. Provided by slurm when using BMI cluster. Leave empty for testing. Defaults to ./",
)
parser.add_argument(
    "--OutputDir",
    type=str,
    nargs="?",
    default="./output",
    help="Sets output directory. Provided by slurm when using BMI cluster. Leave empty for testing. Defaults to ./output",
)
parser.add_argument(
    "--DataDir",
    type=str,
    nargs="?",
    default="./dataset",
    help="Sets datasets directory. Provided by slurm when using BMI cluster. Leave empty for testing. Defaults to ./dataset",
)
parser.add_argument(
    "--Tasks",
    type=int,
    nargs="?",
    default=4,
    help="Number of Tasks. Can be used to set expected resources in pytorch. Defaults to 8",
)
parser.add_argument(
    "--CPUSPerTask",
    type=int,
    nargs="?",
    default=2,
    help="Number of allocated CPUs per Task (threads). Can be used to set expected resources in pytorch. Defaults to 8",
)
parser.add_argument(
    "--GPUS",
    type=int,
    nargs="?",
    default=1,
    help="Number of allocated GPUs. Can be used to set expected resources in pytorch. Defaults to 1",
)
parser.add_argument(
    "--Mem",
    type=str,
    nargs="?",
    default="32G",
    help="Memory allocated to slurm job. Can be used to set expected resources in pytorch. Defaults to 32000 [MB]",
)
parser.add_argument(
    "--UserArguments",
    type=str,
    nargs="?",
    default=None,
    help="User arguments passed via the parameters field in the WeS3 UI",
)

# Pass arguments into variables
args = parser.parse_args()

job_dir = args.ScratchDir
download_dir = args.DownloadDir
output_dir = args.OutputDir
#datasets_dir = args.DataDir
datasets_dir = os.path.join(download_dir, 'dataset')
tasks = args.Tasks
cpus_per_task = args.CPUSPerTask
gpus = args.GPUS
mem = args.Mem
user_args = args.UserArguments

# Creating missing directories

os.makedirs(datasets_dir, exist_ok=True)

###---YOUR CODE FROM HERE---###

print(f"The job directory is:\t\t\t{job_dir}", flush=True)
print(f"The download directory is:\t\t{download_dir}", flush=True)
print(f"The output directory is:\t\t{output_dir}", flush=True)
print(f"The datasets directory is:\t\t{datasets_dir}", flush=True)
print(f"The number tasks set is:\t\t{tasks}", flush=True)
print(f"The number of CPUs per task is:\t\t{cpus_per_task}", flush=True)
print(f"The number of GPUs available is:\t{gpus}", flush=True)
print(f"The amount of RAM available is:\t\t{mem}", flush=True)
print(f"The arguments passed via WeS3:\t\t{user_args}", flush=True)


os.makedirs(os.path.join(datasets_dir, 'raw'), exist_ok=True)
os.makedirs(os.path.join(datasets_dir, 'interim'), exist_ok=True)
os.makedirs(os.path.join(datasets_dir, 'processed'), exist_ok=True)
print("Directories ensured: ", os.path.join(datasets_dir, 'raw'), os.path.join(datasets_dir, 'interim'), os.path.join(datasets_dir, 'processed'))

def print_directory_structure(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(f"Directory: {dirpath}")
        
        if dirnames:
            print("Subdirectories:")
            for dirname in dirnames:
                print(f" - {dirname}")
        else:
            print("No subdirectories.")
        
        if filenames:
            print("Files:")
            for filename in filenames:
                print(f" - {filename}")
        else:
            print("No files found.")
        
        print("\n")

# Define directories for processing the data
raw_data_dirs = [
    os.path.join(datasets_dir, "raw/Task03_Liver/imagesTr"),
    os.path.join(datasets_dir, "raw/Task03_Liver/imagesTs"),
    os.path.join(datasets_dir, "raw/Task08_HepaticVessel/imagesTr"),
    os.path.join(datasets_dir, "raw/Task08_HepaticVessel/imagesTs")
]

processed_data_dirs = [
    os.path.join(datasets_dir, "processed/Task03_Liver/imagesTr"),
    os.path.join(datasets_dir, "processed/Task03_Liver/imagesTs"),
    os.path.join(datasets_dir, "processed/Task08_HepaticVessel/imagesTr"),
    os.path.join(datasets_dir, "processed/Task08_HepaticVessel/imagesTs")
]



flag_dir = os.path.join(datasets_dir, 'flag.txt')


# Call functions to download data and process it
if check_flag_status(flag_dir) == False:
    download_and_prepare_data(download_dir, os.path.join(datasets_dir, 'raw'))
    #process_all_data(raw_data_dirs, processed_data_dirs)
    set_flag_status(flag_dir)
else:
    print('Data already processed')


print_directory_structure(download_dir)

pre_model_path = os.path.join(output_dir, "liver_model_real.pth")
liver_labels_dir = os.path.join(datasets_dir,"raw/Task03_Liver/labelsTr")
liver_train_dir = os.path.join(datasets_dir,"raw/Task03_Liver/imagesTr")

vessels_labels_dir = os.path.join(datasets_dir,"raw/Task08_HepaticVessel/labelsTr")
vessels_train_dir = os.path.join(datasets_dir,"processed/Task08_HepaticVessel/imagesTr")

liver_model_save_path = os.path.join(output_dir, "liver_model_real3.pth")
vessel_model_save_path = os.path.join(output_dir, "vessel_model.pth")
train_model(liver_train_dir, liver_labels_dir, liver_model_save_path, batch_size=192, num_epochs=15, learning_rate=1e-4, preloaded_model_path=None) 

