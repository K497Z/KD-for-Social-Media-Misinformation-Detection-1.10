# KD-for-Social-Media-Misinformation-Detection
This project implements a machine learning framework based on multi-teacher knowledge distillation, primarily applied to fake news detection tasks using the Weibo21 dataset. By introducing an agent policy from reinforcement learning, the model dynamically evaluates and fuses the logits of multiple pre-trained teacher models (such as BERT and RoBERTa). This fused knowledge is then distilled and taught to student models with fewer parameters (such as Chinese-RoBERTa), thereby maximizing the accuracy of fake news detection and classification while maintaining efficient inference.

Core Features:
**Multi-Teacher Knowledge Distillation:** The system simultaneously integrates BERT and RoBERTa as teacher models to extract text features, providing richer soft-label knowledge.

**Dynamic Agent Policy:** A multilayer perceptron (MLP) is designed as the agent weight allocator. It receives the outputs of student and multiple teacher models as input and dynamically calculates and assigns importance weights to different teacher models during the distillation process at different training stages.

**Dynamic Agent Policy:** A multilayer perceptron (MLP) is designed as the agent weight allocator. It receives the outputs of student and multiple teacher models as input and dynamically calculates and assigns importance weights to different teacher models during the distillation process at different stages of training. Extreme Performance Optimization: The code has been deeply optimized for A40 GPU environments, enabling TF32 matrix multiplication support, AMP (Automatic Mixed Precision) training, and PyTorch memory fragmentation optimization (expandable_segments: True), allowing the model to support batch sizes up to 580.

Comprehensive Experimentation and Validation Mechanisms: A built-in 5-fold cross-validation (5-Fold CV) framework is included, along with rich ablation experiments and hyperparameter sensitivity analysis scripts.

Project Structure and Core File Description: This project contains complete code logic from data reading, model training, validation evaluation to various controlled ablation experiments:

`train_teachers_init.py`: Used to initialize pre-trained teacher models (BERT and RoBERTa), train the teacher network using cross-entropy loss, and save the optimal weights for validation metrics to `best_teacher.pt`.

`run_main_ours.py` & `run_5fold_final.py`: The core training scripts of the project. These scripts include a complete student model training process, agent-based dynamic weight allocation logic, and the calculation of KL divergence distillation loss. `run_5fold_final.py` performs standard five-fold cross-validation and saves the validation statistics for each fold.

`run_5fold_ablation_no_agent.py`: Agent ablation experiment script. Removes the dynamic weight agent module, downgrading to a simple static averaging method to fuse the outputs of the two teacher models.

`run_5fold_ablation_input.py`: Unprivileged input ablation experiment script. Forces the model to read only news content (`news_content`) when building the dataset, excluding additional features such as comments, to ensure fairness in benchmark testing under unprivileged information.

`run_single_teacher.py`: Single-teacher baseline experiment script. Used to evaluate the model performance when using only BERT or only RoBERTa to guide the student model.

`run_sensitivity.py`: Hyperparameter sensitivity analysis tool. This section includes comparative experiments on different numerical iterations of key parameters for knowledge distillation: Temperature and Alpha weights in the loss function.

`model.py`: Encapsulates the basic network structure definitions for the Teacher Model and the Student Model (integrating dynamic feature calculation).

`evaluation.py`: The model validation and evaluation module. It encapsulates the calculation logic for detailed evaluation metrics, providing calculations for important parameters such as Macro F1, Accuracy, and AUC area.

`trainer.py` & `trainer_twstd.py`: Provides basic batch training (Trainer) wrappers for the model, including combined Loss calculation logic based on cross-entropy and KL divergence, and a learning rate scheduler to optimize the step size.

`utils.py`: Contains utility functions for unified setting of random number seeds, basic loss function definition, and feature aggregation.

Evaluation Metrics
The project evaluation used comprehensive, multi-dimensional metrics to examine the performance of the detection model:

Overall Performance Metrics: Macro F1 score, overall accuracy, and AUC.

Detailed Classification: Output independent Precision, Recall, and F1 scores for Fake and Real news respectively.

Quick Start Guide
Data Preparation: First, ensure the text dataset (weibo21.csv) is placed in the data/weibo21/ path specified in the code configuration.

Training the Teacher Network: Run the `train_teachers_init.py` script to train and save the weight parameters of the best-performing BERT and RoBERTa teacher models respectively.

Performing Core Distillation Training: Execute `run_5fold_final.py` to start the multi-teacher knowledge distillation process based on dynamic agents. The final metric results after the experiment will be serialized and saved to `5fold_results_final.json`, and the detailed process log will be output to a text log file.
