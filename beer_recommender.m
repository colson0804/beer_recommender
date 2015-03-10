function beer_recommender()

[D, b] = load_data();
%%% run perceptron for 3 initial points %%%

% Calculate fixed steplength - via Lipschitz constant (see Chap 9 for 
% explanation) - for use in all three runs
lam = 10^2;        % regularization parameter 
L = 2*norm(diag(b)*D')^2;
alpha = 1/(L + 2*lam);        % step length
x0 = [1;2;3;4];    % initial point
x = grad_descent_soft_SVM(D,b,x0,alpha,lam);

% Run perceptron second time
lam = 10;
alpha = 1/(L + 2*lam);        % step length
y = grad_descent_soft_SVM(D,b,x0,alpha,lam);

% Run perceptron third time
lam = 10^-2;
alpha = 1/(L + 2*lam);        % step length
z = grad_descent_soft_SVM(D,b,x0,alpha,lam);


%%% gradient descent function for perceptron %%%
function x = grad_descent_soft_SVM(D,b,x0,alpha,lam)
    % Initializations 
    x = x0;
    iter = 1;
    max_its = 3000;
    grad = 1;

    while  norm(grad) > 10^-6 && iter < max_its
        
        % form gradient and take step
        grad = find_grad(D, b, x, alpha, lam); 
        x = x - alpha*grad;

        % update iteration count
        iter = iter + 1;
    end
end

%%% plots everything %%%
function plot_all(A,b,x,y,z)
    
    % plot points 
    ind = find(b == 1);
    scatter3(A(ind,2),A(ind,3), A(ind, 4), 'Linewidth',2,'Markeredgecolor','b','markerFacecolor','none');
    hold on
    ind = find(b == -1);
    scatter3(A(ind,2),A(ind,3), A(ind, 4), 'Linewidth',2,'Markeredgecolor','r','markerFacecolor','none');
    hold on

    % plot separators
    s =[min(A(:,2)):.01:max(A(:,2))];
    plot (s,(-x(1)-x(2)-x(3)*s)/x(4),'k','linewidth', 2);
    hold on

    plot (s,(-y(1)-y(2)-y(3)*s)/y(4),'g','linewidth', 2);
    hold on

    plot (s,(-z(1)-z(2)-z(3)*s)/z(4),'m','linewidth', 2);
    hold on

    set(gcf,'color','w');
    axis([ (min(A(:,2)) - 1) (max(A(:,2)) + 1) (min(A(:,3)) - 1) (max(A(:,3)) + 1) (min(A(:,4)) - 1) (max(A(:, 4)) - 1)])
    box off
    
    % graph info labels
    xlabel('a_1','Fontsize',14)
    ylabel('a_2  ','Fontsize',14)
    zlabel('a_3', 'Fontsize', 14)
    set(get(gca,'YLabel'),'Rotation',0)

end

%%% loads data %%%
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


    A = [ones(size(data, 1), 1), data(:, 2), data(:, 3), data(:, 4)]';
    b = data(:, end);
end

function grad = find_grad(D, b, x, alpha, lam)
    N = numel(b);
    k = numel(x);
    in = ones(N, 1) - diag(b)*D'*x;
    in = max(0, in(:));
    U = eye(k);
    U(1,1) = 0;
    grad = -2*D*diag(b)*in + 2*lam*U*x;
end
end