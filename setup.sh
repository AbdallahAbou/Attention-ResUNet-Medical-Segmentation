# Author:    FName LName
# Created:   DD.MM.YYYY
#
# Description
#
# Example call from slurm_job_template
# bash setup.sh /job /download /output /datasets
##############################

JOB_DIR=${1:-"./"}
DOWNLOAD_DIR=${2:-"./"}
OUTPUT_DIR=${3:-"./output"}
DATASETS_DIR=${4:-"./dataset"}

mkdir $JOB_DIR $DOWNLOAD_DIR $OUTPUT_DIR $DATASETS_DIR

###--- YOUR CODE FROM HERE ---###

mkdir -p $DATASETS_DIR/raw
mkdir -p $DATASETS_DIR/interim
mkdir -p $DATASETS_DIR/processed


echo "Hello World!"
echo "The job directory is $JOB_DIR"
echo "The downoad directory is $DOWNLOAD_DIR"
echo "The output directory is $OUTPUT_DIR"
echo "The datasets directory is $DATASETS_DIR"
