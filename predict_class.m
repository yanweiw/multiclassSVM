%% predict testing data class with alpha vector 

function prediction_vec = predict_class(alpha_vec, x_mat, y_vec, test_data, poly_deg)

        % find support vector because we cannot find an explicit form
        % for w vector of the decision boundary due to non-linear kernel 
        
        support_index = alpha_vec > 0.0001; % smaller than 0.0001 are assumed to be zeros
        support_mat_x = x_mat(support_index,:);
        support_vec_y = y_vec(support_index);
        support_alpha = alpha_vec(support_index);
        
        % test new data points
        M = size(support_vec_y,1); % size of support vectors
        b = 1/M * sum(support_vec_y - ((support_mat_x * support_mat_x').^poly_deg * ...
            (support_vec_y .* support_alpha)));
        prediction_vec = (test_data * support_mat_x').^poly_deg * (support_vec_y .* support_alpha) + b;
        
end