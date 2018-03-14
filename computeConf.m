%% Compute confusion matrix and post-processing

function conf_mat = computeConf(votes, test_samples_labels) 

fprintf('\n\nTraining and testing end, compute confusion_matrix...\n\n');

conf_mat = zeros(10,10); 
[max_counts, max_index] = max(votes,[],2);
for i = 1:size(max_index, 1)
    conf_mat(test_samples_labels(i) + 1, max_index(i)) ...
        = conf_mat(test_samples_labels(i) + 1, max_index(i)) + 1;
end

disp(conf_mat);
accuracy = trace(conf_mat) / size(test_samples_labels,1);
fprintf('Accuracy is: %.3f\n\n',accuracy);

end