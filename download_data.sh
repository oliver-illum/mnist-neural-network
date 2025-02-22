#!/bin/bash

#this command terminates the script if any commands fail
set -e

DATA_DIR="data"
mkdir -p "${DATA_DIR}"

BASE_URL="https://ossci-datasets.s3.amazonaws.com/mnist"
FILES=(
  "train-images-idx3-ubyte.gz"
  "train-labels-idx1-ubyte.gz"
  "t10k-images-idx3-ubyte.gz"
  "t10k-labels-idx1-ubyte.gz"
)

for FILE in "${FILES[@]}"; do
  URL="${BASE_URL}/${FILE}"
  echo "Downloading ${URL}..."
  wget -q --show-progress -P "${DATA_DIR}" "${URL}"
done

echo "Extracting files..."
for GZ_FILE in "${DATA_DIR}"/*.gz; do
  gunzip -f "${GZ_FILE}"
done

echo "MNIST dataset downloaded and extracted in the '${DATA_DIR}' directory."

