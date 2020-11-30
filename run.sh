#!/bin/bash

echo "Running code for Exercise 2."

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist create it!"
  mkdir plots
fi

echo "Creating the output directory if it does not exist"
if [ ! -d "output" ]; then
  echo "Directory does not exist create it!"
  mkdir output
fi

echo "Creating the data directory if it does not exist"
if [ ! -d "data" ]; then
  echo "Directory does not exist create it!"
  mkdir data
fi


echo "Downloading Data"
wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m11.txt
wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m12.txt
wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m13.txt
wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m14.txt
wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m15.txt

mv satgals* data/

echo "Creating virtual environment and install dependencies."
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt

echo "Run the first problem ..."
python3 -W ignore code/problem1.py


echo "Run the second problem ..."
python3 -W ignore code/problem2.py

# Run twice to prevent undefined references
echo "Generating the pdf"
pdflatex -interaction=batchmode -jobname=solutions main.tex
pdflatex -interaction=batchmode -jobname=solutions main.tex
pdflatex -interaction=batchmode -jobname=solutions main.tex

echo "All done! Results saved to solutions.pdf."