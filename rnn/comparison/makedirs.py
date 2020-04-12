import os


def makedirs(example, date):
    #date = datetime.date(datetime.now())

    date_folder = "ex{}/{}".format(example, str(date))
    try:
        os.mkdir(date_folder)
    except:
        print("Folder '{}' already exists.".format(date_folder))

    subfolders = ["mse", 
                  "nmse", 
                  "networks",
                  "plots",
                  "train_losses",
                  "true_values",
                  "matlab_exp"]

    for sf in subfolders:
        try:
            subfolder = "{}/{}".format(date_folder, sf)
            os.mkdir(subfolder)
        except:
            print("Subfolder '{}' already exists.".format(subfolder))

