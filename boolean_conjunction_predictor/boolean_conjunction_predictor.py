import numpy as np
import csv


# hypothesis prediction
def predict(row, all_negative_hypothesis_numerics):
    prediction = 1

    # doubling row to mach all_negative_hypothesis_numerics
    temp = []
    for x, y in zip(row, row):
        temp.append(x)
        temp.append(y)

    # placing parameters relevant to hypothesis
    for a, b in zip(temp, all_negative_hypothesis_numerics):
        if b != -1:
            if b == 1:
                prediction = prediction * a
            elif b == 0:
                prediction = prediction * int(not a)

    return prediction


training_examples = np.loadtxt("data.txt")
# length of matrix -1
num_of_parameters = len(training_examples[0, :]) - 1
# matrix separation
x = training_examples[:, 0: num_of_parameters]
y = training_examples[:, num_of_parameters:num_of_parameters+1]

# preparing all negative hypothesis
all_negative_hypothesis_strings = []  # for output
all_negative_hypothesis_numerics = []  # for use in algorithm
for i in range(1, num_of_parameters+1):
    all_negative_hypothesis_strings.append("x"+str(i))
    all_negative_hypothesis_numerics.append(1)
    all_negative_hypothesis_strings.append("not(x"+str(i)+")")
    all_negative_hypothesis_numerics.append(0)

# row iteration
y_index = 0  # index for y vector
for row in x:
    # if y from example in current row is 1
    if y[y_index] == 1:
        # Placing parameters in hypothesis
        prediction = predict(row, all_negative_hypothesis_numerics)

        if prediction == 0:
            index = 0  # for iterate through all_negative_hypothesis_numerics
            # Improving hypothesis
            for i in row:
                if i == 1:
                    all_negative_hypothesis_numerics[index + 1] = -1
                elif i == 0:
                    all_negative_hypothesis_numerics[index] = -1
                index += 2
    y_index += 1

# if all_negative_hypothesis_numerics index == -1 then don't take this index from all_negative_hypothesis_strings
output_list = []
for (a, b) in zip(all_negative_hypothesis_numerics, all_negative_hypothesis_strings):
    if a != -1:
        output_list.append(b)

# writing to answer to file
with open('output.txt', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    writer.writerow(output_list)

