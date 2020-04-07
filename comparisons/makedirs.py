import os
#from datetime import datetime


def makedirs(example, date):
    #date = datetime.date(datetime.now())

    date_folder = "ex{}/{}".format(example, str(date))
    os.mkdir(date_folder)

    subfolders = ["mse", 
                  "nmse", 
                  "networks",
                  "plots",
                  "train_losses",
                  "true_values",
                  "matlab_exp"]

    for sf in subfolders:
        subfolder = "{}/{}".format(date_folder, sf)
        os.mkdir(subfolder)
