import os


def makedirs(date):
    #date = datetime.date(datetime.now())

    date_folder = "{}".format(str(date))
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
                  "trajectories",]

    subsubfolders = ["train", "test"]

    for sf in subfolders:
        try:
            subfolder = "{}/{}".format(date_folder, sf)
            os.mkdir(subfolder)
        except:
            print("Subfolder '{}' already exists.".format(subfolder))

        if sf == "trajectories":
            for ssf in subsubfolders:
                try:
                    subsubfolder = "{}/{}/{}".format(date_folder, sf, ssf)
                    os.mkdir(subsubfolder)
                except:
                    print("Subfolder '{}' already exists.".format(subsubfolder))


