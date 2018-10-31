import numpy as np 
import math
import random

class node:
    def __init__(self):
        self.branches = []
        self.value = None
        self.feature_id = -1
        self.threshold = -1
        self.tree_id = 1
        self.node_id = 0
        self.gain = 0
    
    def feed(self, x):
        if (self.value != None):
            return self.value
        else:
            if x[self.feature_id] < self.threshold:
                return self.branches[0].feed(x)
            else:
                return self.branches[1].feed(x)

    def print_self(self):
        print("tree={:2d}, node={:3d}, feature={:2d}, thr={:6.2f}, gain={:f}".format(self.tree_id, self.node_id, self.feature_id, self.threshold, self.gain))

def breadth_first_print(tree):
    queue = []

    n = tree
    queue.append(n)
    
    while len(queue) != 0:
        front = queue.pop(0)
        front.print_self()

        if len(front.branches) >= 1:
            queue.append(front.branches[0])

        if len(front.branches) == 2:
            queue.append(front.branches[1])

    for node in queue:
        node.print_self()



def decision_tree(training_file, test_file, option, pruning_thr) :
    training_data = read_file(training_file)
    #training_labels = [x[-1] for x in training_data]
    #training_data = [x[:-1] for x in training_data]
    test_data = read_file(test_file)
    test_labels = [x[-1] for x in test_data]
    test_data = [x[:-1] for x in test_data]

    if option == "optimized":
        tr = DTL_TopLevel(training_data, pruning_thr)
        breadth_first_print(tr)

        print(len(tr.branches), tr.feature_id, tr.threshold)
        acc = 0
        for i in range(len(test_labels)):
            out = tr.feed(test_data[i])
            if isinstance(out, dict):
                out = argmax(out)
            this_acc = 0
            if out == test_labels[i]:
                acc += 1
                this_acc = 1
            
            print("ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}".format(i+1, int(out), int(test_labels[i]), this_acc))
        print("classification accuracy={:6.4f}".format(acc / len(test_labels)))
    elif option == "randomized":
        tr = DTL_TopLevel(training_data, pruning_thr, True)
        breadth_first_print(tr)

        print(len(tr.branches), tr.feature_id, tr.threshold)
        acc = 0
        for i in range(len(test_labels)):
            out = tr.feed(test_data[i])
            if isinstance(out, dict):
                out = argmax(out)
            this_acc = 0
            if out == test_labels[i]:
                acc += 1
                this_acc = 1
            
            print("ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}".format(i+1, int(out), int(test_labels[i]), this_acc))
        print("classification accuracy={:6.4f}".format(acc / len(test_labels)))
    elif option == "forest3":
        forest = [DTL_TopLevel(training_data, pruning_thr, True) for i in range(3)]

        acc = 0
        for i in range(len(test_labels)):
            out = [tr.feed(test_data[i]) for tr in forest] 
            best_out = []
            for o in out:
                if isinstance(o, dict):
                    o = argmax(o)
                best_out.append(o)
            out = max(set(best_out), key=best_out.count)

            this_acc = 0
            if out == test_labels[i]:
                acc += 1
                this_acc = 1
            
            print("ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}".format(i+1, int(out), int(test_labels[i]), this_acc))
        print("classification accuracy={:6.4f}".format(acc / len(test_labels)))
    elif option == "forest15":
        forest = [DTL_TopLevel(training_data, pruning_thr, True) for i in range(15)]
        acc = 0
        for i in range(len(test_labels)):
            out = [tr.feed(test_data[i]) for tr in forest] 
            best_out = []
            for o in out:
                if isinstance(o, dict):
                    o = argmax(o)
                best_out.append(o)
            out = max(set(best_out), key=best_out.count)

            this_acc = 0
            if out == test_labels[i]:
                acc += 1
                this_acc = 1
            
            print("ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}".format(i+1, int(out), int(test_labels[i]), this_acc))
        print("classification accuracy={:6.4f}".format(acc / len(test_labels)))
def distribution(labels):
    total = len(labels)
    return {attr: labels.count(attr) / total for attr in set(labels)}

def argmax(distribution):
    #please send this a dictionary s.t. class : percent
    m = (-1, 0)
    for v in distribution:
        if distribution[v] > m[1]:
            m = (v, distribution[v])
    
    return m[0]

def DTL_TopLevel(examples, pruning_thr, randomized = False):
    attributes = {i : i for i in range(len(examples[0]) - 1)}
    training_labels = [x[-1] for x in examples]
    # Same as dist(examples) but easier
    default = distribution(training_labels)
    return DTL(examples, attributes, default, pruning_thr, randomized)

tree_id = 1
node_id = 1
def DTL(examples, attrs, default, pruning_thr, randomized):
    global tree_id, node_id
    # No examples here, return the most common from last row
    if len(examples) < pruning_thr:
        n = node()
        n.node_id = node_id
        node_id += 1
        n.value = default
        return n
    # All examples same label, return that label
    elif all_equal(examples):
        n = node()
        n.node_id = node_id
        node_id += 1
        n.value = examples[0][-1]
        return n
    # No defaults, calculate the next split
    else:
        # split based on the first attribute for now
        if randomized:
            (best_attr, best_thr, best_gain) = choose_attribute_randomized(attrs, examples)
        else:
            (best_attr, best_thr, best_gain) = choose_attribute_optimized(attrs, examples)
        # create a new tree
        n = node()
        n.tree_id = tree_id
        n.node_id = node_id
        n.gain = best_gain
        node_id += 1
        # Set the nodes threshold
        n.threshold = best_thr
        # Set the nodes attribute to split based on 
        # (relies on best_attr to be a unique value 1 - # of classes)
        n.feature_id = attrs[best_attr]
        next_default = distribution([x[-1] for x in examples])
        examples_left = [x for x in examples if x[best_attr] < best_thr]
        examples_right = [x for x in examples if x[best_attr] >= best_thr]
        n.branches = [DTL(examples_left, attrs, next_default, pruning_thr, randomized), DTL(examples_right, attrs, next_default, pruning_thr, randomized)]

        return n

def choose_attribute_optimized(attrs, examples):
    max_gain = -1
    best_attr = -1
    best_thr = -1

    for attr in attrs.values():
        attr_values = [ex[attr] for ex in examples]
        L = min(attr_values)
        M = max(attr_values)

        for k in range(1, 51):
            thr = L + k * (M-L) / 51
            gain = information_gain(examples, attr, thr)
            if gain > max_gain:
                max_gain = gain
                best_attr = attr
                best_thr = thr
    return (best_attr, best_thr, max_gain)

def choose_attribute_randomized(attrs, examples):
    max_gain = -1
    best_attr = -1
    best_thr = -1

    attr = random.choice(attrs)

    attr_values = [ex[attr] for ex in examples]
    L = min(attr_values)
    M = max(attr_values)

    for k in range(1, 51):
        thr = L + k * (M-L) / 51
        gain = information_gain(examples, attr, thr)
        if gain > max_gain:
            max_gain = gain
            best_attr = attr
            best_thr = thr
    
    return (best_attr, best_thr, max_gain)

def information_gain(examples, attr, threshold):
    left = [ex for ex in examples if ex[attr] < threshold]
    right = [ex for ex in examples if ex[attr] >= threshold]
    base_ent = calculate_entropy(distribution([x[-1] for x in examples]).values())
    left_ent = calculate_entropy(distribution([x[-1] for x in left]).values())
    right_ent = calculate_entropy(distribution([x[-1] for x in right]).values())
    return base_ent - ((len(left) / len(examples)) * left_ent) - ((len(right) / len(examples)) * right_ent)

def all_equal(A):
    if len(A) == 0:
        return True
    #if a single element is given, return true
    if not isinstance(A[0], list):
        return True
    #otherwise check if all are equal to the first label
    return len(set([a[-1] for a in A])) <= 1

def read_file(fname):
    lines = []
    with open(fname) as f:
        for line in f:
            lines.append(list(map(float, line.split())))

    return lines

def calculate_entropy(k):
    ent = 0
    for ki in k:
        ent += ki * math.log2(ki)
    return -ent
#option = one of {optimized, randomized, forest3, forest15}
decision_tree("pendigits_training.txt", "pendigits_test.txt", "forest15", 50)