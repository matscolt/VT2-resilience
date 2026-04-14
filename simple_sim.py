#This is going to be a simple simulation of the FESTO line.
#There is going to be a selection of modules and there process times based on the recorded data.

#There is an input and output csv file

#import packages
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

#----Constants within the line
date = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
inputfile = f"input.csv"
outputfile = f"output_{date}.csv"
with open(inputfile, "r", newline="") as infile:
    reader = list(csv.reader(infile))

#time runing the sim in seconds
time = reader[4][2]
timestep = 0.1

#----Define modules and process times


#----Define the logic of the line



#----Write results to a file


#----Main loop
def main():
    print("Main loop")
    print(f"Output file is {outputfile}")
    print(time)

if __name__ == "__main__":
    main()