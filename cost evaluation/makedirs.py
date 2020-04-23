import os
#from datetime import datetime


def makedirs(example, date):
    #date = datetime.date(datetime.now())

    date_folder = "ex{}/{}".format(example, str(date))
    try:
        os.mkdir(date_folder)
    except:
        print("Folder '{}' already exists.".format(date_folder))

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
        try:
            subfolder = "{}/{}".format(date_folder, sf)
            os.mkdir(subfolder)
        except:
            print("Subfolder '{}' already exists.".format(subfolder))
