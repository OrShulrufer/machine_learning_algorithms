import librosa, numpy, glob
from scipy.stats import stats
import os


def get_mfccs(path, label=-1):
    files = librosa.util.find_files(path)
    files = numpy.asarray(files)
    mfcc_list = []

    for file in files:
        y, sr = librosa.load(file, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfcc = stats.zscore(mfcc, axis=1)

        if label != -1:
            mfcc_list.append((mfcc,label))
        else:
            mfcc_list.append((mfcc,file))

    return mfcc_list


# Locate num_neighbors with are most closest
def get_neighbors(train, test_row, num_neighbors):
    # Go through  all train objects and store distances to test object
    distances = list()
    for train_row in train:
        dist = numpy.linalg.norm(test_row-train_row[0])
        distances.append((train_row, dist))

    # sort ascending and get first num_neighbors
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])

    return neighbors


# Getting one test object and series of train object to predict hom
# the test object is more alike from train objects
def prediction(train, test_row, num_neighbors):

    neighbors = get_neighbors(train, test_row, num_neighbors)

    output_values = [row[-1] for row in neighbors]

    prediction = max(set(output_values), key=output_values.count)

    return prediction



# Get all train data
train_mfcc1 = get_mfccs("./train_data/one/", 1)
train_mfcc2 = get_mfccs("./train_data/two/", 2)
train_mfcc3 = get_mfccs("./train_data/three/", 3)
train_mfcc4 = get_mfccs("./train_data/four/", 4)
train_mfcc5 = get_mfccs("./train_data/five/", 5)

# Gather all train data lists in one list
all_train_mfcc = train_mfcc1 + train_mfcc2 + train_mfcc3 + train_mfcc4 + train_mfcc5

# getting test list mfcc
all_test_mfcc = get_mfccs("./test_files/")


file = open("output.txt", "w")
# Go through all test data one by one and predict: one, two, three, four or five
# And write it down on a output file
for row in all_test_mfcc:
    file.write("%s - %d\n" % (os.path.basename(row[1]), prediction(all_train_mfcc, row[0], 1)))
file.close()






