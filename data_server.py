#Module to load dataset and divide it into train, test and validation sets for X and Y

def Load_dataset(filename):

    import numpy as np

    try:
        file = open(filename)
    except:
        print("Unable to open the file")
    else:
        print("File opened")

    X_list = []
    Y_list = []

    first_line = True

    print("Preparing dataset...")

    for line in file:
        if first_line == True:          #Skipping first line of file which contains name of data columns
            first_line = False
        else:
            row = line.split(',')
            Y_list.append(int(row[0]))
            # Pixel values for each picture are saved in csv in single cells as a string containing values splited by
            # space-bar so we need to split that string to substrings, convert them to ints and add these pixel values to list one by one
            X_list.append([int(pixel) for pixel in row[1].split()])

    X_data = np.array(X_list)
    Y_data = np.array(Y_list)

    file.close()

    training_sample = 28709
    test_sample = 3589
    validation_sample = 3589

    #Divide whole data set to train, test and validation data sets for X and Y
    X_train = X_data[0:training_sample, :]
    X_test = X_data[training_sample:training_sample + test_sample, :]
    X_validate = X_data[training_sample + test_sample:training_sample + test_sample + validation_sample, :]

    Y_train = Y_data[0:training_sample]
    Y_validate = Y_data[training_sample:training_sample + test_sample]
    Y_test = Y_data[training_sample + test_sample:training_sample + test_sample + validation_sample]

    print("Dataset prepared")

    return X_train, Y_train, X_test, Y_test, X_validate, Y_validate


#Showing single picture from X_data numpy array, number of picture is controled by variable 'which_one'
def Show_Picture(wchich_one, X_data):

    import matplotlib.pyplot as plt

    plt.imshow(X_data[wchich_one,:].reshape(48, 48))
    plt.show()

    return None