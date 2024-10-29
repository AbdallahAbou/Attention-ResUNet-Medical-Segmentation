# Author:    FName LName
# Created:   DD.MM.YYYY
#
# Description
#
# Example call from slurm_job_template
# bash setup.sh /job /download /output /datasets
##############################

JOB_DIR=${1:-"./job"}
DOWNLOAD_DIR=${2:-"./download"}
OUTPUT_DIR=${3:-"./output"}

mkdir $JOB_DIR $DOWNLOAD_DIR $OUTPUT_DIR 

###--- YOUR CODE FROM HERE ---###


echo "Hello World!"
echo "The job directory is $JOB_DIR"
echo "The downoad directory is $DOWNLOAD_DIR"
echo "The output directory is $OUTPUT_DIR"
