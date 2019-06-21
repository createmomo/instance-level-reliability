import pandas as pd

def preprocessing_input(instances, crowd_labels, labels_for_initialisation, expert_labels):
    # instances
    instances_array = pd.read_csv(instances, header=None).values

    # crowd labels
    crowd_labels_array = pd.read_csv(crowd_labels, header=None).values

    # labels for initialisation
    labels_for_initialisation_array = None
    if (labels_for_initialisation is None) == False:
        labels_for_initialisation_array = pd.read_csv(labels_for_initialisation, header=None).values

    # expert labels
    expert_labels_array = None
    if (expert_labels is None) == False:
        expert_labels_array = pd.read_csv(expert_labels, header=None).values

    return instances_array, crowd_labels_array, labels_for_initialisation_array, expert_labels_array

def prepare_input_for_model(shuffled_indices, instances, crowd_labels, labels_for_initialisation, expert_labels):
    current_instances = instances[shuffled_indices]
    current_crowd_labels = crowd_labels[shuffled_indices]

    current_labels_for_initialisation = None
    current_expert_labels = None

    if labels_for_initialisation is not None:
        current_labels_for_initialisation = labels_for_initialisation[shuffled_indices]
    if expert_labels is not None:
        current_expert_labels = expert_labels[shuffled_indices]

    return current_instances, current_crowd_labels, current_labels_for_initialisation, current_expert_labels