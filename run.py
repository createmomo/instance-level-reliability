import argparse
import logging

import torch
import torch.optim as optim
import pandas as pd
from sklearn import metrics

from models import MLP, Classifier, Model
from utils import preprocessing_input, prepare_input_for_model

def main(args):
    # --- prepare logging ---
    logger = logging.getLogger('aggregation')
    formatter = logging.Formatter('%(message)s')
    fh = logging.FileHandler(args.log)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # --- preprocessing input ---
    instances, crowd_labels, labels_for_initialisation, expert_labels = preprocessing_input(args.instances,
                                                                                            args.crowd_labels,
                                                                                            args.labels_for_initialisation,
                                                                                            args.expert_labels)
    num_instances, feature_dimension = instances.shape
    _, num_annotators = crowd_labels.shape
    num_labels = args.num_labels

    instances = torch.from_numpy(instances).float()
    crowd_labels = torch.from_numpy(crowd_labels).float()

    if labels_for_initialisation is not None:
        labels_for_initialisation = torch.from_numpy(labels_for_initialisation).squeeze(-1)
    if expert_labels is not None:
        expert_labels = torch.from_numpy(expert_labels).squeeze(-1)

    if num_labels > 2:
        evaluation_metric = 'binary'
    else:
        evaluation_metric = 'macro'

    log_message = '{} instances; {} annotators; {} labels; {} dimension;'.format(num_instances,
                                                                                 num_annotators,
                                                                                 num_labels,
                                                                                 feature_dimension)
    logger.warning(args)
    print(log_message)
    logger.warning(log_message)

    # --- init classifier and reliability estimator ---
    classifier = Classifier(MLP(feature_dimension, args.num_hidden_units_classifier, num_labels))
    reliability_estimator = Classifier(MLP(args.num_hidden_units_classifier + num_annotators,
                                           args.num_hidden_units_reliability_estimator,
                                           2))

    # --- init our model ---
    model = Model(classifier, reliability_estimator)

    # --- init optimiser ---
    optimiser = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # check learnable parameters
    print('\n[parameters]')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # --- pre-train (initialise) the model if the labels for initialisation is provided
    if args.labels_for_initialisation is not None:
        for epoch in range(args.num_initialise_epochs):
            # prepare input data into tensor
            shuffled_indices = torch.randperm(num_instances)
            current_instances, current_crowd_labels, current_labels_for_initialisation, current_expert_labels = prepare_input_for_model(shuffled_indices,
                                                                                                                 instances,
                                                                                                                 crowd_labels,
                                                                                                                 labels_for_initialisation,
                                                                                                                 expert_labels)

            optimiser.zero_grad()
            loss, predictions = model.compute_pre_train_loss(current_instances,
                                                current_crowd_labels,
                                                current_labels_for_initialisation,
                                                num_instances, num_annotators,
                                                args.num_hidden_units_classifier)

            # evaluate the classifier performance on the labels for initialisation
            if epoch % 50 == 0:
                classifier_prediction = torch.argmax(predictions, dim=-1).tolist()
                train_f1 = metrics.f1_score(current_labels_for_initialisation.tolist(),
                                            classifier_prediction,
                                            average=evaluation_metric)
                message = 'pre-train-classifier-epoch-{}\tf1:{:.02f}'.format(epoch,train_f1)
                print(message)
                logger.warning(message)

                if train_f1 >= args.performance_threshold_of_initialisation:
                    break

            loss.backward()
            optimiser.step()

    # --- START TO TRAIN THE MODEL ---
    for outer_epoch in range(args.num_outer_epochs):
        # prepare input data into tensor
        shuffled_indices = torch.randperm(num_instances)
        current_instances, current_crowd_labels, current_labels_for_initialisation, current_expert_labels = prepare_input_for_model(
            shuffled_indices,
            instances,
            crowd_labels,
            labels_for_initialisation,
            expert_labels)

        # only update the classifier
        for param in model.reliability_estimator.parameters():
            param.requires_grad = False

        for inner_epoch in range(args.num_inner_epochs):
            optimiser.zero_grad()

            # compute margin here
            gold_label_posterior, reliability_posterior, predictions, reliability_scores, crowd_nan_mask, _ = model.compute_posteriors(current_instances,
                                                                                               current_crowd_labels,
                                                                                               num_instances,
                                                                                               num_annotators,
                                                                                               num_labels,
                                                                                               args.num_hidden_units_classifier)

            # classifier cross entropy loss
            classifier_cross_entropy_loss = - torch.sum(torch.eye(num_labels)[torch.argmax(gold_label_posterior,dim=-1)] * torch.log(predictions + 1e-9)) / num_instances
            loss = classifier_cross_entropy_loss

            loss.backward()
            optimiser.step()

        for param in model.reliability_estimator.parameters():
            param.requires_grad = True

        # only update the reliability estimator
        for param in model.classifier.parameters():
            param.requires_grad = False

        for inner_epoch in range(args.num_inner_epochs):
            optimiser.zero_grad()

            # compute margin here
            gold_label_posterior, reliability_posterior, predictions, reliability_scores, crowd_nan_mask, y_reliability_scores = model.compute_posteriors(current_instances,
                                                                                               current_crowd_labels,
                                                                                               num_instances,
                                                                                               num_annotators,
                                                                                               num_labels,
                                                                                               args.num_hidden_units_classifier)

            # reliability estimator cross entropy loss
            total_size = torch.sum(crowd_nan_mask)
            current_estimated_gold_labels = torch.argmax(gold_label_posterior,dim=-1)
            is_reliable = torch.eq(current_crowd_labels, current_estimated_gold_labels.unsqueeze(-1).float()).long()  # shape: num_instances * num_annotators
            reliability_estimator_loss = model.reliability_estimator_cross_entropy_loss(y_reliability_scores.view(-1, 2),
                                                                                       is_reliable.view(-1))
            reliability_estimator_loss *= crowd_nan_mask.view(-1)
            reliability_estimator_cross_entropy_loss = reliability_estimator_loss.sum() / total_size

            loss = reliability_estimator_cross_entropy_loss

            loss.backward()
            optimiser.step()

        for param in model.classifier.parameters():
            param.requires_grad = True

        if expert_labels is not None:
            with torch.no_grad():
                # compute margin here
                gold_label_posterior, reliability_posterior, \
                predictions, reliability_scores, crowd_nan_mask, _ = model.compute_posteriors(current_instances,
                                                                                           current_crowd_labels,
                                                                                           num_instances,
                                                                                           num_annotators,
                                                                                           num_labels,
                                                                                           args.num_hidden_units_classifier)

                estimated_gold_labels = torch.argmax(gold_label_posterior, dim=-1).tolist()

                precision = metrics.precision_score(current_expert_labels.tolist(), estimated_gold_labels,average=evaluation_metric)
                recall = metrics.recall_score(current_expert_labels.tolist(), estimated_gold_labels, average=evaluation_metric)
                f1 = metrics.f1_score(current_expert_labels.tolist(), estimated_gold_labels, average=evaluation_metric)

                message = 'epoch:{}\tprecision:{:.03f}\trecall:{:.03f}\tf1:{:.03f}'.format(outer_epoch, precision, recall, f1)
                print(message)
                logger.warning(message)

    # --- output the estimated gold labels and per-instance reliabilities ---
    with torch.no_grad():
        # compute margin here
        gold_label_posterior, reliability_posterior, \
        predictions, reliability_scores, crowd_nan_mask, _ = model.compute_posteriors(instances,
                                                                                      crowd_labels,
                                                                                      num_instances,
                                                                                      num_annotators,
                                                                                      num_labels,
                                                                                      args.num_hidden_units_classifier)

        estimated_gold_labels = torch.argmax(gold_label_posterior, dim=-1).tolist()
        estimated_reliability_scores = reliability_posterior[:,:,1].tolist() # the column 1 is the posterior probability of that when the annotator is reliable; 0 is when unreliable

        pd.DataFrame(estimated_gold_labels).to_csv(args.output_predicted_labels, index=False, header=None)
        pd.DataFrame(estimated_reliability_scores).to_csv(args.output_reliabilities, index=False, header=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parse args')

    # --- input paths ---
    parser.add_argument('-ii','--instances', type=str, default='demo_instances.txt')
    parser.add_argument('-ic','--crowd-labels', type=str, default='demo_crowd_labels.txt')
    parser.add_argument('-nl','--num-labels', type=int, default=2)

    # provide the labels for initialisation if you can (strongly recommend)
    parser.add_argument('-il','--labels-for-initialisation', type=str, default='demo_labels_for_initialisation.txt')

    # provide the expert labels for evaluation if you have it
    parser.add_argument('-ie','--expert-labels', type=str, default='demo_expert_labels.txt')

    # --- output paths ---
    parser.add_argument('-ol','--output-predicted-labels',type=str, default='demo_predicted_labels.output')
    parser.add_argument('-or','--output-reliabilities', type=str, default='demo_reliabilities.output')
    parser.add_argument('-log','--log', type=str, default='demo_log.txt')

    # --- model hyper-parameters ---
    parser.add_argument('-lr','--learning-rate', type=float, default=0.005)
    parser.add_argument('-wc','--weight-clipping', type=float, default=5.0)
    parser.add_argument('-wd','--weight-decay', type=float, default=0.001)

    parser.add_argument('-hc','--num-hidden-units-classifier', type=int, default=10)
    parser.add_argument('-hr','--num-hidden-units-reliability-estimator', type=int, default=5)

    parser.add_argument('-ep','--num-initialise-epochs', type=int, default=500)
    parser.add_argument('-pt','--performance-threshold-of-initialisation', type=float, default=0.95)
    parser.add_argument('-ei','--num-inner-epochs', type=int, default=20)
    parser.add_argument('-eo','--num-outer-epochs', type=int, default=5)

    args = parser.parse_args()
    main(args)