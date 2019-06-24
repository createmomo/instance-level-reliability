## Modelling Instance-Level Annotator Reliability for Natural Language Labelling Tasks

### [Introduction]
The **probabilistic model** incorporates **neural networks** to model the dependency between latent variables and instances. It can simultaneously estimate **per-instance annotator reliability** and the **correct labels** for natural language labelling tasks. 

It is re-implemented by using PyTorch. Please cite the paper when using the code:
* Li, M., Myrman, A.F., Mu, T. and Ananiadou, S., 2019, June. [Modelling Instance-Level Annotator Reliability for Natural Language Labelling Tasks](https://www.aclweb.org/anthology/N19-1295). In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ (pp. 2873-2883).

_**Motivation**_ 
Annotators (e.g., experts or crowd workers) may provide **labels of inconsistent quality** due to their varying expertise and reliability in a domain; Previous studies have mostly focused on estimating **overall** or **per-category reliability**; In practice, the **reliability** of an annotator may depend on **each specific instance**.

_**What task can the code be used for?**_ 
_Examples including, but not limited to_: 
* Learning a model from **noisy labels** produced by multiple annotators or data sources; 
* Estimate the **reliability of annotators** across different instances for **spammer/unreliable annotator/data source detection**, **low-quality label reduction**, **task allocation** in [pro-active learning](https://www.aclweb.org/anthology/W17-2314) or crowdsourcing environment (i.e., select the most appropriate annotators to request the labels for each instance);
* Predict the **correct label** for each instance.

### [Requirements]
* Python 3.7+
* Pandas 0.24.2+
* Scikit-learn 0.20.3+
* PyTorch 1.0+

### [To Prepare for Running]
The **crowd annotations** and **instances** are required. The **expert (i.e., gold) labels** and the **labels for initialisating (i.e., pre-train)** the model are optional.

_**Crowd Annotation (e.g., demo_crowd_labels.txt):**_
Each line includes the crowd labels from multiple annotators for one instance, and these crowd labels are separated by commas. For example,
`1,1,1,0,0`. 
If the crowd labels are not complete, in other words, not all the instances were annotated by all the annotators, the missing labels can be left as empty as follows: `1,1,,0,0`

_**Instances (e.g., demo_instances.txt):**_
Each line is the feature representation of each instance, e.g., the vector representation of one text fragment or image.

_**Expert/Gold Labels (e.g., demo_expert_labels.txt):**_
Each line is the expert's/gold label of each instance. If the expert label is provided, the code will evaluate the performance (i.e., precision, recall and f1-score) of predicting the true labels at each epoch.

_**Labels for Initialisation or Pre-Train the model (e.g., demo_labels_for_initialisation.txt):**_
Each line is a label for each instance. These labels will be used to pre-train/initialise our model. The labels could be obtained by applying other methods (e.g., arbitrarily select one label from annotators' annotations, Majority Voting, [Dawid-Skene](https://github.com/dallascard/dawid_skene) or [MACE](https://www.isi.edu/publications/licensed-sw/mace/)) to the crowd annotations. It is strongly recommended that preparing the labels for initialisation. Although these labels are not perfect, our method can still learn some useful information from them for a better starting point than random parameter initialisation.

### [To Run the Code]
There are 2 ways to set the hyper-parameters. 

One way is to modify the hyper-parameter default values in the code (i.e., from line 212, run.py), then run command: `python run.py`. 

Another way is that specifying the values in the command, e.g.,:
`python run.py -ii demo_instance.txt -ic demo_crowd_labels.txt`

**Note that** you may need to modify the **hidden layer sizes** of the **two neural networks** (e.g., the classifier and reliability estimator) in our model according to your task. For example, if your instance vector is 200-dimensional, you may want the hidden layer of the classifier and reliability estimator to have 50 units (`-hc 50`) and 25 units (`-hr 25`) respectively. You may also need to set the **number of classes** (e.g., `-nl 3` if your label set is {cat, dog, other}) in your dataset.

Below is the detailed description of each hyper-parameter.

| short  | long  | description  | default value or example|
|---|---|---|---|
| -ii| --instances  | Path of instances |demo_instances.txt  |
| -ic  |--crowd-labels | Path of crowd annotations |demo_crowd_labels.txt| 
|-nl|--num-labels|The number of classes in your dataset|2 e.g., 2 classes (cat, dog) in your dataset|
|-il|--labels-for-initialisation| Path of labels for initialisation/pre-training of our model| demo_labels_for_initialisation.txt|
|-ie|--expert-labels|Path of expert/gold labels|demo_expert_labels.txt|
|-ol|--output-predicted-labels| Path of the output of the predicted label. Each line is the predicted label of each instance.|demo_predicted_labels.output|
|-or|--output-reliabilities| Path of estimated per-instance reliability output of each annotator. Each line includes the estimated reliabilities of all the annotators on the current instance. |demo_reliabilities.output| 
|-log|--log|Path of the model running log|demo_log.txt|
|-lr|--learning-rate|Learning rate of the Adam optimiser|0.005|
|-wc|--weight-clipping|The weight clipping threshold of the Adam optimiser|5.0|
|-wd|--weight-decay|The weight decay value of the Adam optimiser|0.001|
|-hc|--num-hidden-units-classifier|The hidden layer size of the classifier|10|
|-hr|--num-hidden-units-reliability-estimator|The hidden layer size of the reliability estimaotr|5|
|-ep|--num-initialise-epochs|The number of epochs for initialisation/pre-training the model|500|
|-pt|--performance-threshold-of-initialisation| The f1-score of the classifier on the labels for initialisation/pre-training. The initialisation/pre-training would be stopped if the classifier performance on the labels for initialisation reaches the value of the maximum number of iterations (i.e. -ep).|0.95|
|-ei|--num-inner-epochs|The number of inner epochs for updating the classifier and the reliability estimator. **If the size of your classifier or reliability estimator is large**, it is best to increase the number.|20|
|-eo|--num-outer-epochs|The number of outer epochs for learning the entire model. We recommend do not set a large value here. A small number such as 5 (i.e., early stopping) is preferred for obtaining a good label prediction performance. The reason for early stopping is explained in 5.5 Training Stability of our [paper](https://www.aclweb.org/anthology/N19-1295).|5|

### [Console Output When Running]
When the code is running, you may get the console output (the output is also recored in the log file) like this:
```
Namespace(crowd_labels='demo_crowd_labels.txt', expert_labels='demo_expert_labels.txt', instances='demo_instances.txt', labels_for_initialisation='demo_labels_for_initialisation.txt', learning_rate=0.005, log='log.txt', num_hidden_units_classifier=10, num_hidden_units_reliability_estimator=5, num_initialise_epochs=500, num_inner_epochs=20, num_labels=2, num_outer_epochs=5, output_predicted_labels='predicted_labels.output', output_reliabilities='reliabilities.output', performance_threshold_of_initialisation=0.95, weight_clipping=5.0, weight_decay=0.001)

1000 instances; 5 annotators; 2 labels; 2 dimension;

pre-train-classifier-epoch-0 f1:0.33
pre-train-classifier-epoch-50 f1:0.61
pre-train-classifier-epoch-100 f1:0.92
pre-train-classifier-epoch-150 f1:0.94
pre-train-classifier-epoch-200 f1:0.95

epoch:0 precision:0.988 recall:0.988 f1:0.988
epoch:1 precision:0.991 recall:0.991 f1:0.991
epoch:2 precision:0.989 recall:0.989 f1:0.989
epoch:3 precision:0.990 recall:0.990 f1:0.990
epoch:4 precision:0.993 recall:0.993 f1:0.993
```

_**Explanation of the output**_:

* `Namespace(...)`: displays the parameter settings.
* `1000 instances; 5 annotators; 2 labels; 2 dimension;`:  describes how many instances, annotators, classes in your dataset and the dimensions of your instance vector.
* `pre-train-classifier-epoch-0 f1:0.33`: indicates the f1-score of the classifier on the labels for initialisation (not on the expert/gold labels). Please find more details about the performance in the description of parameter `-pt`.
*  `epoch:4 precision:0.993 recall:0.993 f1:0.993`: if you provided the expert/gold labels, indicates the performance (i.e., the precision, recall and f1-score) on these labels of the current outer epoch.

### [Final Output]
2 output files will be produced once the running is finished.

_**Predicted Correct Labels (e.g., demo_predicted_labels.output):**_
Each line is the predicted label for each instance.

_**Estimated Per-Instance Reliabilities of Each Annotator (e.g., demo_reliabilities.output):**_ 
Each line includes the estimated annotators' per-instance reliabilities on the current instance. Therefore, this output would have the number of instances lines and the number of annotators columns.

### [FAQ - Frequently Asked Questions]
**Q: Is there any difference between this re-implemented version and the original model described in the paper?**

**A:** This code implements the cross-entropy (training alternatingly) method for training our model. This method achieved much better and more stable performance than the models learned using EM training reported in our paper. In the cross-entropy loss function, instead of minimising the cross-entropy between the posteriors and output of classifier/reliability estimator, the estimated labels obtained according to the posteriors are used to compute the cross-entropy loss. Because in this way, the model could obtain good results faster than that of the original model. But you can still get good performance if you use the loss function described by Equation 6 and 7 in our paper.


****Q:** Some times the model obtained the highest label prediction performance at the first outer epoch, but it eventually starts to decrease at the following epochs. For example, the f1-scores at the first beginning epochs could be "0.989->0.987->0.984->0.977->0.987...".**

**A:** Please refer to ***Early Stopping in Section 5.5 Training Stability*** of our paper for this situation. In fact, the label prediction performance in the first epoch has always been pretty good. If you prefer a faster model when using the code, you can just set the number of outer epochs to 1. 
