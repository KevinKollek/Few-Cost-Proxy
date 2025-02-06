import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json


class Net(nn.Module):
    def __init__(self, input_shape=3072):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_shape, 3072)
        self.fc2 = nn.Linear(3072, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x.flatten()))
        x = self.fc2(x)
        return F.softmax(x)


# Define the number of classes and examples per class
num_classes = 10
examples_per_class = 5000
examples_per_class_test = 1000

# Define the dimensions of the images
image_shape = (32, 32, 3)

# Define the number of training and test examples
num_train_examples = num_classes * examples_per_class
num_test_examples = num_classes * examples_per_class_test

# Define maximum count of each class
max_count = (num_train_examples) // num_classes
max_count_test = (num_test_examples) // num_classes

# Generate the random Gaussian images
train_labels = []
test_labels = []
num_networks = 10

# with open('synthetic_cifar10.json', 'r') as f:
#     # Load the JSON data from the file
#     data = json.load(f)

networks = []
for i in range(num_networks):
    networks.append(Net())

# Define counters for each class
class_counts = {i: 0 for i in range(num_classes)}
class_counts_test = {i: 0 for i in range(num_classes)}

# Initialize train_images as an empty list
train_images = []
test_images = []

while any(count < max_count_test for count in class_counts_test.values()):
    # Generate a new set of random Gaussian images
    new_image = torch.from_numpy(np.random.randn(*image_shape)).float()

    output = []
    for i in range(num_networks):
        output.append(networks[i](new_image))

    max_values = []
    max_indices = []
    for tensor in output:
        max_val, max_idx = torch.max(tensor, dim=0)
        max_values.append(max_val.item())
        max_indices.append(max_idx.item())

    # find the maximum value and index across all tensors
    global_max_value, global_max_index = torch.max(torch.tensor(max_values), dim=0)
    global_max_index = max_indices[global_max_index.item()]

    if class_counts[global_max_index] < examples_per_class:
        train_images.append(new_image)
        train_labels.append(global_max_index)
        class_counts[global_max_index] += 1
    elif class_counts_test[global_max_index] < examples_per_class_test:
        test_images.append(new_image)
        test_labels.append(global_max_index)
        class_counts_test[global_max_index] += 1

# Assign labels to each image in the training and test sets
# train_labels = np.array([assign_label(image) for image in train_images])
# test_labels = np.array([assign_label(image) for image in test_images])


train_labels = torch.tensor(train_labels, dtype=torch.float)
test_labels = torch.tensor(test_labels, dtype=torch.float)

# train_images = torch.stack(train_images).numpy().tolist()
# test_images = torch.stack(test_images).numpy().tolist()
# train_labels = train_labels.detach().cpu().numpy().tolist()
# test_labels = test_labels.detach().cpu().numpy().tolist()

train_data = {"image": train_images, "label": train_labels}
test_data = {"image": test_images, "label": test_labels}

data = {"train": train_data, "test": test_data}

torch.save(data, 'synthetic_cifar10.pt')
