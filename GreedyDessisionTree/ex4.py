from functools import reduce
import math
from graphviz import Digraph

# Tree class for decision tree ----------------------------------------------------------------------------------------


class Forest:
    def __init__(self, train_set, val_set, attr_set):
        self.trees = dict()
        self.make_trees_list(train_set, val_set, attr_set)

    def make_trees_list(self, train_set, val_set, attr_set):
        # Builds the forest by making a dictionary to include all possible trees according to their depth
        # epsilon = depth of current tree
        max_depth_decision_tree = Tree()
        max_depth_decision_tree.decision_tree_build(train_set, attr_set)
        epsilon = 0

        n = Tree.number_of_nodes
        current_number_of_nodes = 0
        while current_number_of_nodes < n:
            Tree.number_of_nodes = 0
            enable_all_features(attributes)

            current_decision_tree = Tree()
            current_decision_tree.decision_tree_build(train_set, attr_set, epsilon)
            current_accuracy = current_decision_tree.accuracy(val_set)
            current_number_of_nodes = Tree.number_of_nodes

            self.trees[epsilon] = current_decision_tree, current_accuracy
            epsilon += 1

        self.trees[epsilon] = max_depth_decision_tree, max_depth_decision_tree.accuracy(val_set)

    def find_max_accuracy(self):
        max_accuracy = 0
        for tree_tuple in self.trees.values():
            if tree_tuple[1] > max_accuracy:
                max_accuracy = tree_tuple[1]
        return max_accuracy

    def find_optimal_tree(self):
        # Finds the optimal tree by accuracy
        max_accuracy = self.find_max_accuracy()
        optimal_tree = None
        for tree_tuple in self.trees.values():
            if tree_tuple[1] == max_accuracy:
                optimal_tree = tree_tuple[0]
                return optimal_tree
        return None


class Tree:
    number_of_nodes = 0
    leaf_index = 0
    visual_model = Digraph(comment='Decision Tree')

    def __init__(self, id=None, value=None, left=None, right=None):
        self.id = id
        self.value = value
        self.left = left
        self.right = right
        self.feature_index = None
        self.left_subset_division = None
        self.right_subset_division = None

    def visual_model_build(self):
        # Build the visual model
        Tree.visual_model.node(self.id, self.value)
        if self.left:
            self.left.visual_model_build()
            if self.left.id is not None:
                self.visual_model.edge(self.id, self.left.id)
        if self.right:
            self.right.visual_model_build()
            if self.right.id is not None:
                self.visual_model.edge(self.id, self.right.id)
        if self.left and self.right:
            with Tree.visual_model.subgraph() as s:
                s.attr(rank='same')
                s.node(self.left.id)
                s.node(self.right.id)

    def draw(self):
        # Draw the visual model
        Tree.visual_model.graph_attr['rankdir'] = 'TB'
        self.visual_model.render('test-output/round-table.gv', view=True)

    def set_subset_division(self, feature, subset):
        # Set the binary split of a given feature in decision tree
        self.left_subset_division = complement_subset(feature, subset)
        self.right_subset_division = subset

    def predict(self, sample):
        # Predict a given sample by decision tree
        if self.left is None and self.right is None:  # Leaf
            return self.value[0:1]
        current_feature_index = self.feature_index
        left_split = self.left_subset_division

        if sample[current_feature_index] in left_split:
            return self.left.predict(sample)
        else:
            return self.right.predict(sample)

    def accuracy(self, test_samples):
        predicted_right = 0
        for i in range(len(test_samples)):
            if str(test_samples[i][1]) is str(self.predict(test_samples[i][0])):
                predicted_right += 1
        return predicted_right / len(test_samples)

    def decision_tree_build(self, training_set, features, eps=None):
        Tree.number_of_nodes += 1
        current_features = [feature.get_attribute() for feature in features if feature.available]
        optimal_subset, optimal_ig, left_child, right_child, optimal_feature = choose_best_feature(current_features, training_set)
        labels_parent = [tup[1] for tup in training_set]
        labels_left = [tup[1] for tup in left_child]
        labels_right = [tup[1] for tup in right_child]
        current_entropy = entropy(labels_parent)
        current_information_gain = information_gain(labels_parent, (labels_right, labels_left))

        if len(current_features) == 0 or optimal_ig == 0 or current_entropy == 0 or eps == 0:  # Leaf
            self.id = 'L' + str(Tree.leaf_index)
            self.value = node_string(current_entropy, current_information_gain, labels_parent, True, self.id)
            Tree.leaf_index += 1
            return

        i = index_of_available_feature(features, optimal_feature)
        self.feature_index = i
        self.id = str(Tree.leaf_index)
        str_parent = 'A' + str(i + 1)
        self.value = node_string(current_entropy, current_information_gain, labels_parent, False, str_parent)
        Tree.leaf_index += 1
        self.set_subset_division(optimal_feature, optimal_subset)
        disable_feature(features, optimal_feature)

        left_features = [Feature(feature.get_attribute(), feature.available) for feature in features]
        right_features = [Feature(feature.get_attribute(), feature.available) for feature in features]
        self.left = Tree()
        self.right = Tree()
        if eps is None:
            self.left.decision_tree_build(left_child, left_features)  # Left feature
            self.right.decision_tree_build(right_child, right_features)  # Right feature
        else:
            self.left.decision_tree_build(left_child, left_features, eps-1)  # Left feature
            self.right.decision_tree_build(right_child, right_features, eps-1)  # Right feature

# Feature class -------------------------------------------------------------------------------------------------------


class Feature:
    def __init__(self, attribute, boolean):
        self.attribute = attribute
        self.available = boolean  # For checking if a given feature is still available to split

    def get_attribute(self):
        return self.attribute


def disable_feature(features, given_feature):
    for feature in features:
        if feature.get_attribute() == given_feature and feature.available:
            feature.available = False
            break


def enable_all_features(features):
    for feature in features:
        feature.available = True


def index_in_given_attribute(features, given_feature):
    i = 0
    for feature in features:
        if feature.get_attribute() == given_feature:
            return i
        i += 1
    return -1


def index_of_available_feature(features, given_feature):
    # Returns index of a given *available* feature in a given features list
    i = 0
    for feature in features:
        if feature.available:
            if reduce(lambda t, j: t and j, map(lambda m, k: m == k, feature.get_attribute(), given_feature), True):
                return i
        i += 1
    return i


def load_attributes():
    attributes_ = []
    file = open("train.txt", "r")
    lines = file.readlines()
    for line in lines:
        if line[0] == "#":
            a = line.split(',')
            a[len(a)-1] = (a[len(a)-1])[0:len((a[len(a)-1]))-1]
            attributes_.append(Feature(a[2:len(a)], True))

    file.close()
    return attributes_


def load_samples(file):
    training_samples = []
    lines = file.readlines()

    for current_line in lines:
        first_char = current_line[0]
        if not (first_char == "%" or first_char == "#" or first_char == "/"):
            current_line = current_line[0:len(current_line)-1]
            current_line_as_list = current_line.split(',')
            label = current_line_as_list[len(current_line_as_list) - 1]
            current_line_as_list.remove(label)
            attributes_ = current_line_as_list
            training_samples.append((attributes_, label))

    file.close()
    return training_samples

# Help Functions ------------------------------------------------------------------------------------------------------


def entropy(feature_labels):
    # feature_labels = all labels of a feature, represented by a tree node
    if len(feature_labels) == 0:
        # To avoid division by zero in probability calculating \
        return 0

    good_labels_counter = 0
    bad_labels_counter = 0

    for example in feature_labels:
        if example == 'G':
            good_labels_counter += 1
        else:
            bad_labels_counter += 1

    if good_labels_counter == 0 or bad_labels_counter == 0:  # In both cases entropy = 0
        return 0

    good_probability = good_labels_counter / (good_labels_counter + bad_labels_counter)
    bad_probability = bad_labels_counter / (good_labels_counter + bad_labels_counter)

    return -(good_probability * math.log(good_probability, 2) + bad_probability * math.log(bad_probability, 2))


def information_gain(father_labels, children_labels):
    # children_labels is a tuple represents labels of left\right children
    father_entropy = entropy(father_labels)
    child1_entropy = entropy(children_labels[0])
    child2_entropy = entropy(children_labels[1])

    if len(father_labels) == 0:
        # There are no labels to split, then stop by informing that this information gain is not informative ( = 0)
        return 0

    # The weights of children_labels entropy (it's prior probability regarding father labels)
    weight1 = len(children_labels[0]) / len(father_labels)
    weight2 = len(children_labels[1]) / len(father_labels)

    return father_entropy - (weight1*child1_entropy + weight2*child2_entropy)


def choose_best_feature(current_features, node):
    # current_features is the set of features left to pick, in the current moment
    # node is an entry in a decision tree containing labels
    if len(node) == 0 or len(current_features) == 0:
        # Stop, because there are no features to choose or because there are no samples to split
        return [], 0, [], [], None

    optimal_info_gain = 0
    optimal_subset = []
    optimal_left = []
    optimal_right = []
    optimal_feature = current_features[0]

    for i in range(len(current_features)):
        info_gain, subset, right, left = split_feature_optimally(current_features[i], node)
        if info_gain >= optimal_info_gain:
            optimal_info_gain = info_gain
            optimal_subset = subset
            optimal_left = left
            optimal_right = right
            optimal_feature = current_features[i]

    return optimal_subset, optimal_info_gain, optimal_left, optimal_right, optimal_feature


def split_feature_optimally(a, node):
    # a is one attribute (feature) from set of attributes A
    # node is an entry in a decision tree containing labels
    global attributes  # Set of attributes
    flag = False  # True -> sample belongs to current_left_child, False -> sample belongs to current_right_child
    max_information_gain = 0
    optimal_subset = []
    optimal_left_child = []
    optimal_right_child = []

    subsets = powerset(a)
    for subset in subsets:
        current_left_child, current_right_child = [], []
        if len(subset) == 0:
            current_left_child = []
            current_right_child = node
        else:
            for sample in node:
                for element in subset:
                    if element == sample[0][index_in_given_attribute(attributes, a)]:
                        flag = True
                if flag:
                    current_left_child.append(sample)
                else:
                    current_right_child.append(sample)
                flag = False

        ig = information_gain([tup[1] for tup in node],
                              [[tup[1] for tup in current_left_child], [tup[1] for tup in current_right_child]])

        if max_information_gain < ig:
            max_information_gain = ig
            optimal_subset = subset
            optimal_left_child = current_left_child
            optimal_right_child = current_right_child

    return max_information_gain, optimal_subset, optimal_left_child, optimal_right_child


def complement_subset(world, given_set):
    # Calculates the complement ser of a given set
    return [val for val in world if val not in given_set]


def powerset(given_set):
    # Calculates the powerset of a given set
    size = len(given_set)
    power_set = list()
    for i in range(1 << size):
        power_set.append([given_set[j] for j in range(size) if (i & (1 << j))])

    return power_set


def find_best_option(labels_in_node):
    count_good_labels = labels_in_node.count('G')
    count_bad_labels = labels_in_node.count('B')
    if count_good_labels < count_bad_labels:
        return 'B'
    return 'G'


def node_string(entropy_, ig, labels, is_leaf, str_key=None):
    string = ''
    if not is_leaf:
        string = str_key + '\n' + 'Entropy: ' + str(entropy_)[:6] + '\nIG: ' + \
                 str(ig)[:6] + '\nGood:' + str(len([label for label in labels if label == 'G'])) + \
                 '\nBad:' + str(len([label for label in labels if label == 'B']))
    else:
        string = str(find_best_option(labels)) + '\n' + 'Entropy: ' + str(entropy_)[:6] + \
                '\nGood:' + str(len([label for label in labels if label == 'G'])) + \
                '\nBad:' + str(len([label for label in labels if label == 'B']))
    return string


# main code -----------------------------------------------------------------------------------------------------------

training_set = load_samples(open("train.txt", "r"))
validation_set = load_samples(open("val.txt", "r"))
test_set = load_samples(open("test.txt", "r"))
attributes = load_attributes()

forest = Forest(training_set, validation_set, attributes)
optimal_tree = forest.find_optimal_tree()
optimal_tree.visual_model_build()
optimal_tree.draw()

print("Accuracy on validation samples is " + str(optimal_tree.accuracy(validation_set))[0:4])
print("Accuracy on test samples is " + str(optimal_tree.accuracy(test_set))[0:4])





