function perceptron_demo_hw()
% perceptron_demo_wrapper - runs the perceptron model on a separable two 
% class dataset consisting of two dimensional data features. 
% The perceptron is run 3 times with 3 initial points to show the 
% recovery of different separating boundaries.  All points and recovered
% as well as boundaries are then visualized.

%%% load data %%%
[D,b] = load_data();

%%% run perceptron for 3 initial points %%%

% Calculate fixed steplength - via Lipschitz constant (see Chap 9 for 
% explanation) - for use in all three runs
L = 2*norm(diag(b)*D')^2;
alpha = 1/L;        % step length

% Run perceptron first time
x0 = [1;2;3];    % initial point
x = grad_descent_perceptron(D,b,x0,alpha);

% Run perceptron second time
x0 = 10*[1;2;3];    % initial point
y = grad_descent_perceptron(D,b,x0,alpha);

% Run perceptron third time
x0 = 100*[1;2;3];    % initial point
z = grad_descent_perceptron(D,b,x0,alpha);

%%% plot everything, pts and lines %%%
plot_all(D',b,x,y,z);


%%% gradient descent function for perceptron %%%
function x = grad_descent_perceptron(D,b,x0,alpha)
    % Initializations 
    x = x0;
    iter = 1;
    max_its = 3000;
    grad = 1;

    while  norm(grad) > 10^-6 && iter < max_its
        
        % form gradient and take step
        grad = ;            % your code goes here!
        x = x - alpha*grad;

        % update iteration count
        iter = iter + 1;
    end
end

%%% plots everything %%%
function plot_all(A,b,x,y,z)
    
    % plot points 
    ind = find(b == 1);
    scatter(A(ind,2),A(ind,3),'Linewidth',2,'Markeredgecolor','b','markerFacecolor','none');
    hold on
    ind = find(b == -1);
    scatter(A(ind,2),A(ind,3),'Linewidth',2,'Markeredgecolor','r','markerFacecolor','none');
    hold on

    % plot separators
    s =[min(A(:,2)):.01:max(A(:,2))];
    plot (s,(-x(1)-x(2)*s)/x(3),'k','linewidth',2);
    hold on

    plot (s,(-y(1)-y(2)*s)/y(3),'g','linewidth',2);
    hold on

    plot (s,(-z(1)-z(2)*s)/z(3),'m','linewidth',2);
    hold on

    set(gcf,'color','w');
    axis([ (min(A(:,2)) - 1) (max(A(:,2)) + 1) (min(A(:,3)) - 1) (max(A(:,3)) + 1)])
    box off
    
    % graph info labels
    xlabel('a_1','Fontsize',14)
    ylabel('a_2  ','Fontsize',14)
    set(get(gca,'YLabel'),'Rotation',0)
    
end

%%% loads data %%%
function [A,b] = load_data()
    data = load('perceptron_data.mat');
    data = data.data;
    A = data(:,1:3);
    A = A';
    b = data(:,4);
end

end
