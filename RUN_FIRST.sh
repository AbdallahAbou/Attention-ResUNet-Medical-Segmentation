#!/bin/bash

echo "Untracking data folder"
git rm -r --cached data

echo "Untracking output folder"
git rm -r --cached output

echo "Removing this script"
rm RUN_FIRST.sh
