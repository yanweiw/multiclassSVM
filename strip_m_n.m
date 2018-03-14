%% function to strip out two class data only for current training in 1-1 scheme

function [x_mat, y_vec] = strip_m_n(data, label, m, n)

x_mat = [];
y_vec = [];
for i = 1:size(data,1)
    if label(i) == m
        y_vec = [y_vec; 1];
    elseif label(i) == n
        y_vec = [y_vec; -1];
    else
        continue
    end
    x_mat = [x_mat; data(i,:)];
end

end