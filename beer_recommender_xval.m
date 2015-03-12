function [lam,xmin] = beer_recommender_xval()
% Here we illustrate 3-fold cross validation using soft-margin SVM and
% polyonimal features
clear all
debug = 0;
survey = 5;     % Number of beers the user gets asked about

% poly degs to look over
lams = logspace(-3,0,20);

% load data etc.,
% A_rem is data that the user hasn't provided
%   i.e. the pool of beers that we will try to suggest
[A, b, A_rem, b_rem] = load_data(debug, survey);

% split points into 3 equal sized sets and plot
c = split_data(A,b);

% do 3-fold cross-validation on data collected from user
[lam,xmin] = cross_validate(A,b,c,lams);

% From lam and xmin, we can now find a beer to suggest

% Function here

function c = split_data(a,b)
    % split data into 3 equal sized sets
    K = length(b);
    order = randperm(K);
    c = ones(K,1);
    K = round((1/3)*K);
    c(order(K+1:2*K)) =2;
    c(order(2*K+1:end)) = 3;
end
        
function [lam,xmin] = cross_validate(A_orig,b,c,lams)  
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
    end
    test_ave = mean(test_errors');
    [val,j] = min(test_ave);
    
%     % plot best separator for all data
     lam = lams(j);
     xmin = fast_grad_descent_L2_soft_SVM(A',b,lam);
end
   

function [A,b, A_rem, b_rem] = load_data(debug, survey)
    % A, b: Data we are performing cross-validation on
    % A_rem, b_rem: Data that is remaining (i.e. user hasn't tried or been
    %   asked about
    data = importdata('beer_data.csv', ',', 1);
    beerNames = data.textdata;
    data = data.data;
    removeRows=[];
    r=0;
 
    if debug == 0
        data(:, 6) = zeros(size(data, 1), 1);
        i = 1;
        surveyed = [];
        while (i <= survey)
            randomBeer = randi([1, 54], 1, 1);
            % Makes sure that you aren't repeating beers
            while (any(surveyed == randomBeer) == 1)
                randomBeer = randi([1, 54], 1, 1);
            end
        
            choice = questdlg(strcat('Do you like...', beerNames(randomBeer+1),'?'), ...
                'Calculating your taste in beer...', ...
                'Yes','No','Haven''t tried it','Haven''t tried it');
            switch choice
                case 'Yes'
                    data(randomBeer, 6) = 1;
                case 'No'
                    data(randomBeer, 6) = -1;
                case 'Haven''t tried it'
                    data(randomBeer, 6) = 0;
            end
            i = i+1;
            surveyed(end + 1) = randomBeer;
            
        end   
    end
    
    for i = 1:size(data,1)
        if data(i, 6) == 0
          r=r+1;
          removeRows(r,1)=i;
        end
    end
    data_rem = data([removeRows], :);
    data([removeRows],:)=[];

    A = [data(:, 2), data(:, 3), data(:, 4), data(:, 5)];
    A_rem = [data_rem(:,2), data_rem(:,3), data_rem(:,4), data_rem(:,5)];
    b = data(:, end);
    b_rem = data_rem(:, end);
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





