%% function to solve quadratic programming 

function alpha_vec = findAlpha(x_mat, y_vec, poly_deg)
% we will use x = quadprog(H,f,A,b,Aeq,beq):
% minimize 0.5 * x'Hx + f'x where x is variable, A*x <= b, Aeq * x = beq

N = size(y_vec,1); % N data points
% use (x'x)^4 non-linear kernal according to LeCun's MNIST SVM performance
H = ((x_mat * x_mat').^poly_deg) .* (y_vec * y_vec');
f = -ones(N,1); % minus because we want to maximize while quadprog minimizes
A = -eye(N);
b = zeros(N,1);
Aeq = [y_vec'; zeros(N-1,N)]; % A zero matrix where 1st row contains y
beq = zeros(N,1); % such that effectively y_vec' * alpha_vec = 0
options = optimoptions('quadprog',...
    'Algorithm','interior-point-convex','Display','off');
alpha_vec = quadprog(H, f, A, b, Aeq, beq, [],[],[], options);

end