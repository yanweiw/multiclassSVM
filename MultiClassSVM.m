%% Felix Yanwei Wang 
% ONE-ONE & ONE-REST & DAG multiclass SVM classifier implementation for MNIST dataset

%% Section 1: Initialization 
clear; clc;
load('MNIST_data.mat')
polynomial_deg = 4;

%% Section 2: Train Classifier with 1-1 scheme 
% loop through (0 - 9) vs (0 - 9), total of 10 * 9 / 2 = 45 combinations
% ... to train 45 different classifier
round = 0;
votes = zeros(size(test_samples_labels,1),10); 
fprintf('Training SVM with nonlinear kernel to classify MNIST data with 1-1 scheme...\n');

% Train classifier_m_n in 45 rounds
for m = 0 : 8
    for n = m + 1 : 9
        round = round + 1;
        fprintf('\nRound %d, training classifier_%d_%d\n', round, m, n);
        % strip out only m class and n class data
        [x_mat, y_vec] = strip_m_n(train_samples,train_samples_labels,m,n);
        % solve langrangian optimization dual form 
        alpha_vec = findAlpha(x_mat, y_vec, polynomial_deg);
        % make prediction based on alpha vector
        pred_vec = predict_class(alpha_vec,x_mat,y_vec,test_samples, polynomial_deg);
        m_class = pred_vec > 0;
        pred_vec(m_class) = m;
        pred_vec(~m_class) = n;
        % for each test entry try add up votes from 45 classifiers 
        for i = 1:size(pred_vec,1)
            % prediction can be 0 while index > 0
            votes(i, pred_vec(i) + 1) = votes(i, pred_vec(i) + 1) + 1; 
        end
    end
end

% Compute confusion matrix
conf_mat_1_1 = computeConf(votes, test_samples_labels);

%% Section 3 Train Classifier with 1-rest scheme
% create 10 classifiers which each compare one class vs the rest
% label the single class +1 while the other class -1/9
round = 0;
pred_mat = zeros(size(test_samples,1),10); % prediction of 10 classifiers
fprintf('Training SVM with nonlinear kernel to classify MNIST data with 1-rest scheme...\n');

% Train classifier_m 
for m = 0 : 9
    round = round + 1;
    fprintf('\nRound %d, training classifier_%d_rest\n', round - 1, m);
    % divide class m and rest and change their labels
    x_mat = train_samples;
    y_vec = train_samples_labels;
    m_class = y_vec == m;
    y_vec(m_class) = 1;
    y_vec(~m_class) = -1/9; % let negative class has target -1/(K-1)
    % solve langrangian optimization dual form 
    alpha_vec = findAlpha(x_mat, y_vec, polynomial_deg);
    % make prediction based on alpha vector
    pred_mat(:, m + 1) = predict_class(alpha_vec,x_mat,y_vec,test_samples, polynomial_deg);
end

% Compute confusion matrix
conf_mat_1_rest = computeConf(pred_mat ,test_samples_labels);

%% Section 4 Train DAGSVM
% train 45 1-1 classifiers in a DAG tree lateral traversal
% tree has root 0 vs 9 and and leaves 0 : 9 in that order

% Initialization
votes = ones(size(test_samples_labels,1),10); 
% votes are initially all 1s, each iteration we take out one guess by
% marking it zero
fprintf('Training SVM with nonlinear kernel to classify MNIST data with DAGSVM scheme...\n');

% Train classifier_m_n iteratively to update predictions vector
for depth = 1 : 9 % tree depth, depth i has i nodes
    for m = 0 : depth - 1  % ASSUMING m < n
        n = m + (10 - depth); % At m_n node, we train, m_n classifier
        fprintf('\nRound %d, training classifier_%d_%d\n',...
            (1 + depth) * depth / 2 - (depth - 1 - m) , m, n);
        [x_mat, y_vec] = strip_m_n(train_samples,train_samples_labels,m,n);
        alpha_vec = findAlpha(x_mat, y_vec, polynomial_deg);
        pred_vec = predict_class(alpha_vec,x_mat,y_vec,test_samples,polynomial_deg);
        m_class = pred_vec > 0; 
        votes(m_class, n + 1) = 0;
        votes(~m_class, m + 1) = 0;
    end
end

% Compute Confusion Matrix
conf_mat_DAG = computeConf(votes, test_samples_labels);

