import os
#from datetime import datetime


def makedirs(example, date):
    #date = datetime.date(datetime.now())

    date_folder = "ex{}/{}".format(example, str(date))
    os.mkdir(date_folder)

    subfolders = ["true_values", 
                  "train_losses", 
                  "plots", 
                  "test_losses", 
                  "networks",
                  "evaluations",
                  "cost_dicts",
                  "comparisons",
                  "trajectories",
                  "comparison_plots"]

    for sf in subfolders:
        subfolder = "{}/{}".format(date_folder, sf)
        os.mkdir(subfolder)
