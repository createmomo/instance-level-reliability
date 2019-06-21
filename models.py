import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()

    def forward(self, input):
        h1 = self.relu(self.l1(input))
        h2 = self.relu(self.l2(h1))
        output = self.l3(h2)

        return h2, output

class Classifier(nn.Module):
    def __init__(self, predictor):
        super(Classifier, self).__init__()
        self.predictor = predictor

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        h, output = self.predictor(input)

        return h, output, self.softmax(output)

class Model(nn.Module):
    def __init__(self, classifier, reliability_estimator):
        super(Model, self).__init__()
        self.reliability_estimator = reliability_estimator
        self.classifier = classifier

        # cross entropy losses
        self.classifier_cross_entropy_loss = nn.CrossEntropyLoss()
        self.reliability_estimator_cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def compute_pre_train_loss(self, instances, crowd_labels, labels_for_initialisation, num_instances, num_annotators, num_hidden_units_classifier):
        last_hidden_layer_output, before_softmax, predictions = self.classifier(instances)
        # classifier loss part
        classifier_loss = self.classifier_cross_entropy_loss(before_softmax, labels_for_initialisation)

        # reliability estimator loss part
        crowd_nan_mask = ~torch.isnan(crowd_labels)
        crowd_size = crowd_nan_mask.sum()
        crowd_nan_mask = (crowd_nan_mask).float() # shape: num_instances * num_annotators

        crowd_annotators_one_hot_matrix = torch.eye(num_annotators).expand(num_instances,num_annotators,num_annotators)
        concatenated_instances = torch.cat([last_hidden_layer_output.unsqueeze(1).expand(num_instances, num_annotators, num_hidden_units_classifier),
                                            crowd_annotators_one_hot_matrix],dim=-1)
        h_reliability_scores, y_reliability_scores, reliability_scores = self.reliability_estimator(concatenated_instances)

        is_reliable = torch.eq(crowd_labels, labels_for_initialisation.unsqueeze(-1).float()).long() # shape: num_instances * num_annotators

        reliability_estimator_loss = self.reliability_estimator_cross_entropy_loss(y_reliability_scores.view(-1,2),
                                                                                   is_reliable.view(-1))
        reliability_estimator_loss *= crowd_nan_mask.view(-1)
        reliability_estimator_loss = reliability_estimator_loss.sum()/crowd_size

        loss = classifier_loss + reliability_estimator_loss

        return loss, predictions

    def compute_posteriors(self, instances, crowd_labels, num_instances, num_annotators,
                           num_labels,
                           num_hidden_units_classifier):
        # classifier output. Shape of predictions: num_instances * num_labels
        last_hidden_layer_output, before_softmax, predictions = self.classifier(instances)

        # reliability estimator output. Shape of reliability_scores: num_instances * num_annotators * 2
        crowd_nan_mask = ~torch.isnan(crowd_labels) # shape: num_instances * num_annotators
        crowd_size = crowd_nan_mask.sum()
        crowd_nan_mask = (crowd_nan_mask).float()

        crowd_annotators_one_hot_matrix = torch.eye(num_annotators).expand(num_instances, num_annotators,
                                                                           num_annotators)
        concatenated_instances = torch.cat(
            [last_hidden_layer_output.unsqueeze(1).expand(num_instances, num_annotators, num_hidden_units_classifier),
             crowd_annotators_one_hot_matrix], dim=-1)
        h_reliability_scores, y_reliability_scores, reliability_scores = self.reliability_estimator(
            concatenated_instances)

        # compute posteriors
        with torch.no_grad():
            # compute gamma_{ij}^k(t) -- Equation (3). i-th instance, j-th annotator, t the possible labels
            # It should be a matrix num_instances * num_annotators * num_labels

            # if annotators are not reliable. Shape of reliability_scores: num_instances * num_annotators * 2
            gamma_not_reliable = reliability_scores[:,:,0] * (1.0/num_labels)# when r' = 0; num_instances * num_annotators

            # if annotators are reliable
            all_possible_labels = torch.tensor(range(num_labels)).float().expand(num_instances,
                                                                                 num_annotators,
                                                                                 num_labels) # num_instances * num_annotators * num_labels
            comparison_result_with_crowd_labels = (all_possible_labels == crowd_labels.unsqueeze(-1)).float() # num_instances * num_annotators * num_labels
            gamma_reliable = reliability_scores[:,:,1].unsqueeze(-1) * comparison_result_with_crowd_labels # when r' = 1 # num_instances * num_annotators * num_labels

            # gamma_{ij}^k(t)
            gamma = gamma_not_reliable.unsqueeze(-1) + gamma_reliable # r = 0 and 1; num_instances * num_annotators * num_labels
            log_gamma = torch.log(gamma)
            log_gamma *= crowd_nan_mask.unsqueeze(-1) # eliminate annotators who did not provide labels useing mask

            # compute pi_{ij}(t,r) --- Equation (2). i-th instance, j-th annotator, t the possible labels, r 0 or 1 (fallible or reliable)
            # all the other annotators except j --- the last item in Equation (2)
            log_gamma_in_eq2 = log_gamma.unsqueeze(1).expand(num_instances, num_annotators, num_annotators, num_labels)
            log_gamma_in_eq2 = log_gamma_in_eq2.sum(dim=2)
            log_gamma_in_eq2 = log_gamma_in_eq2 - log_gamma # num_instances * num_annotators * num_labels

            temp_log_pi_ij_t_r = log_gamma_in_eq2 + torch.log(predictions.unsqueeze(1))
            # when not reliable pi_{ij}{t,r=0}; num_instances * num_annotators * num_labels
            log_pi_ij_t_r0 = temp_log_pi_ij_t_r + \
                             torch.log(reliability_scores[:,:,0].unsqueeze(-1)) + \
                             torch.log(torch.tensor(1.0/num_labels))

            # when reliable pi_{ij}{t,r=1}; num_instances * num_annotators * num_labels
            log_pi_ij_t_r1 = temp_log_pi_ij_t_r + torch.log(reliability_scores[:,:,1].unsqueeze(-1)) * comparison_result_with_crowd_labels

            # concatenate the results and apply mask
            log_pi_ij_t_r = torch.cat((log_pi_ij_t_r0.unsqueeze(-1),
                                       log_pi_ij_t_r1.unsqueeze(-1)),dim=-1)

            pi_ij_t_r = torch.exp(log_pi_ij_t_r) * crowd_nan_mask.unsqueeze(-1).unsqueeze(-1)

            # compute posterior
            # p(t_i = t | ai, xi) --- Equation (4)
            gold_label_posterior = pi_ij_t_r.sum(dim=-1).sum(dim=1) # Shape: num_instances * num_labels
            gold_label_posterior /= gold_label_posterior.sum(dim=-1, keepdim=True)

            # p(r_ij = r | ai, xi) --- Equation (5)
            reliability_posterior = pi_ij_t_r.sum(dim=2) # Shape: num_instances * num_annotators * 2
            reliability_posterior /= reliability_posterior.sum(dim=-1, keepdim=True)
            reliability_posterior[reliability_posterior != reliability_posterior] = 0.0

        return gold_label_posterior, reliability_posterior, predictions, reliability_scores, crowd_nan_mask, y_reliability_scores







