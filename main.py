"""
Author:    FName LName
Created:   DD.MM.YYYY

Description

Example call
main.py /job /download /output /datasets
"""
import os
import argparse

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
    default="./data",
    help="Sets datasets directory. Provided by slurm when using BMI cluster. Leave empty for testing. Defaults to ./data",
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
datasets_dir = args.DataDir
tasks = args.Tasks
cpus_per_task = args.CPUSPerTask
gpus = args.GPUS
mem = args.Mem
user_args = args.UserArguments

# Creating missing directories
dirs = [job_dir, download_dir, output_dir, datasets_dir]
for dir in dirs:
    os.makedirs(dir, exist_ok=True)

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