function L2reg_cross_validation_classification()
% Here we illustrate 3-fold cross validation using soft-margin SVM and
% polyonimal features
clear all

% poly degs to look over
poly_deg = 10;
lams = logspace(-3,0,20);

% load data etc.,
[A,b] = load_data();

% split points into 3 equal sized sets and plot
c = split_data(A,b);

% do 3-fold cross-validation
cross_validate(A,b,c,poly_deg,lams);  

function c = split_data(a,b)
    % split data into 3 equal sized sets
    K = length(b);
    order = randperm(K);
    c = ones(K,1);
    K = round((1/3)*K);
    c(order(K+1:2*K)) =2;
    c(order(2*K+1:end)) = 3;

    % plot train/test sets for each cross-validation instance
    for j = 1:3
        subplot(2,3,j)
        box on
        plot_pts(a,b,c,j)
    end
end
        
function cross_validate(A_orig,b,c,poly_deg,lams)  
    %%% performs 3-fold cross validation
    % generate features     
    
    A = [];
    for k = 1:poly_deg
        A = [A, A_orig(:,1).^(poly_deg + 1 - k)];
    end
    A = [ones(size(A,1),1), A, A_orig(:,2)];
    
    % solve for weights and collect test errors
    test_errors = [];
    train_errors = [];

    for i = 1:length(lams)
        lam = lams(i);
        test_resids = [];
        train_resids = [];

        for j = 1:3
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
        
        hold on
        subplot(2,3,i)
        plot_poly(x,poly_deg,'m')
        
        %%% find the worst performer (per split) and plot it %%%
        [val,j] = max(test_errors(:,i));
        A_1 = A(find(c ~= i),:);
        b_1 = b(find(c ~= i)); 

        % run soft-margin SVM with chosen lambda
        lam = lams(j);
        x = fast_grad_descent_L2_soft_SVM(A_1',b_1,lam);
            
        hold on
        subplot(2,3,i)
        plot_poly(x,poly_deg,'g')
    end
    test_ave = mean(test_errors');
    [val,j] = min(test_ave);
    
    % plot best separator for all data
    lam = lams(j);
    xmin = fast_grad_descent_L2_soft_SVM(A',b,lam);
    
    hold on
    subplot(2,3,5)
    plot_poly(xmin,poly_deg,'k')
    xlabel('a_1','Fontsize',14,'FontName','cmr10')
    ylabel('a_2','Fontsize',14,'FontName','cmr10')
    box on
    set(gcf,'color','w');

    % plot training and testing errors
    figure(2)
    s = mean(test_errors');
    semilogx(lams,s,'--','MarkerEdgeColor',[1 0.7 0],'MarkerFaceColor',[1 0.7 0])
    hold on
    semilogx(lams,s,'o','MarkerEdgeColor',[1 0.7 0],'MarkerFaceColor',[1 0.7 0])
    hold on
    
    s = mean(train_errors');
    semilogx(lams,s,'--','MarkerEdgeColor',[0.1 0.8 1],'MarkerFaceColor',[0.1 0.8 1])
    hold on
    semilogx(lams,s,'o','MarkerEdgeColor',[0.1 0.8 1],'MarkerFaceColor',[0.1 0.8 1])
    set(gcf,'color','w');
    box on
    xlabel('lambda','Fontsize',14,'FontName','cmr10')
    ylabel('error','Fontsize',14,'FontName','cmr10')
    set(get(gca,'YLabel'),'Rotation',90)
%     set(gca,'YTickLabel',[])
%     set(gca,'YTick',[])
    set(gcf,'color','w');
    set(gca,'FontSize',12); 
    box off
    
    figure(3)
    bar(0:length(xmin)-1,xmin);
    xlabel('x-index','Fontsize',14,'FontName','cmr10')
    set(gcf,'color','w');
    box off
end
    
function plot_poly(x,poly_deg,color)
    % Generate poly seperator
    range = [0:0.01:1];
    x_new = -x/x(end);
    t = 0;
    for i = 1:poly_deg
        t = t + x_new(i+1)*range.^(poly_deg + 1 - i);
    end
    t = t + x_new(1);

    % plot separator
    plot(range,t,color,'linewidth',1.25);
    axis([0 1 -1.5 1.5])
end

function plot_pts(A,b,c,j)
    
    % plot training set
    ind = find(c ~= j);
    ind2 = find(b(ind) == 1);
    ind3 = ind(ind2);
    plot(A(ind3,1),A(ind3,2),'o','MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',7)
    hold on
    ind2 = find(b(ind) == -1);
    ind3 = ind(ind2);
    plot(A(ind3,1),A(ind3,2),'o','MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',7)
    
    % plot test set?
    ind = find(c == j);
    ind2 = find(b(ind) == 1);
    ind3 = ind(ind2);
    plot(A(ind3,1),A(ind3,2),'o','MarkerEdgeColor',[0.75 0.75 1],'MarkerFaceColor',[0.75 0.75 1],'MarkerSize',7)

    hold on
    ind2 = find(b(ind) == -1);
    ind3 = ind(ind2);
    plot(A(ind3,1),A(ind3,2),'o','MarkerEdgeColor',[1 0.75 0.75],'MarkerFaceColor',[1 0.75 0.75],'MarkerSize',7)
end

function [A,b] = load_data()
    data = load('new_sin_class_approx_data.mat');
    data = data.data;
    A = data(:,1:end - 1);
    b = data(:,end);

    % Generate points and plot
    subplot(2,3,5)
    ind = find(b == 1);
    plot(A(ind,1),A(ind,2),'o','MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',7)
    hold on
    ind = find(b == -1);
    plot(A(ind,1),A(ind,2),'o','MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',7)
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





