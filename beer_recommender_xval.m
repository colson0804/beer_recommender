function lam = beer_recommender()
% Here we illustrate 3-fold cross validation using soft-margin SVM and
% polyonimal features
clear all

% poly degs to look over
poly_deg = 1;
lams = logspace(-3,0,20);

% load data etc.,
[A,b] = load_data();

% split points into 3 equal sized sets and plot
c = split_data(A,b);

% do 3-fold cross-validation
lam = cross_validate(A,b,c,poly_deg,lams);  

function c = split_data(a,b)
    % split data into 3 equal sized sets
    K = length(b);
    order = randperm(K);
    c = ones(K,1);
    K = round((1/3)*K);
    c(order(K+1:2*K)) =2;
    c(order(2*K+1:end)) = 3;
end
        
function lam = cross_validate(A_orig,b,c,poly_deg,lams)  
    %%% performs 3-fold cross validation
    % generate features     
    
    A = [ones(size(A,1),1), A_orig];
    
    % solve for weights and collect test errors
    test_errors = [];
    train_errors = [];

    for i = 1:length(lams)
        lam = lams(i);
        test_resids = [];
        train_resids = [];

        for j = 1:3     % folds
            A_1 = A(find(c ~= j),:);
            b_1 = b(find(c ~= j));
            A_2 = A(find(c==j),:);
            b_2 = b(find(c==j));
            
            % run soft-margin SVM with chosen lambda
            x = fast_grad_descent_L2_soft_SVM(A_1',b_1,lam);
            
           resid = evaluate(A_2,b_2,x);
           test_resids = [test_resids resid];
           resid = evaluate(A_1,b_1,x);
           train_resids = [train_resids resid];
        end
        test_errors = [test_errors; test_resids];
        train_errors = [train_errors; train_resids];

    end

    % find best parameter per data-split
    for i = 1:3
        [val,j] = min(test_errors(:,i));
        A_1 = A(find(c ~= i),:);
        b_1 = b(find(c ~= i));
        
        % run soft-margin SVM with chosen lambda
        lam = lams(j);
        x = fast_grad_descent_L2_soft_SVM(A_1',b_1,lam);

        % run soft-margin SVM with chosen lambda
        lam = lams(j);
        x = fast_grad_descent_L2_soft_SVM(A_1',b_1,lam);
    end
    test_ave = mean(test_errors');
    [val,j] = min(test_ave);
    
    % plot best separator for all data
    lam = lams(j);
    xmin = fast_grad_descent_L2_soft_SVM(A',b,lam);
end
   

function [A,b] = load_data()
    data = importdata('beer_data.csv', ',', 1);
    data = data.data;
    removeRows=[];
    r=0;

    for i = 1:size(data,1)
        if data(i, 6) == 0
            r=r+1;
            removeRows(r,1)=i;
        end
    end
    data([removeRows],:)=[];


    A = [data(:, 2), data(:, 3), data(:, 4)];   %% add data(:, 4)
    b = data(:, end);
end

function x = fast_grad_descent_L2_soft_SVM(D,b,lam)
    % Initializations 
    x = randn(size(D,1),1);        % initial point
    H = diag(b)*D';
    L = 2*norm(H)^2;           % Lipschitz constant of perceptron
    alpha = 1/(L + 2*lam);       % step length 

    l = ones(size(D,2),1);
    iter = 1;
    max_its = 2000;
    grad = 1;
    y = x;
    while  norm(grad) > 10^-6 && iter < max_its        
        % form gradient and take accelerated step
        x0 = x;
        grad = - 2*H'*max(l-H*y,0) + 2*lam*[0;y(2:end)];
        x = y - alpha*grad;
        y = x + iter/(iter+3)*(x - x0);
        
        % update iteration count
        iter = iter + 1;
    end
end

function score = evaluate(A,b,x)
% compute score of trained model on test data

    score = 0;  % initialize
    s = A*x;
    ind = find(s > 0);
    s(ind) = 1;
    ind = find(s <= 0);
    s(ind) = -1;
    t = s.*b;
    ind = find(t < 0);
    t(ind) = 0;
    score = 1 - sum(t)/numel(t);

end
end




