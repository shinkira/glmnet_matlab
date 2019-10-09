% GlmTutorial_binomial.m

clear
close all

% Here I assume two types of trials -- stim 1: black cue and stim 2: white cue
% A neural population of 100 cells show a larger response to the white cue with added white noise
% Choice (0 or 1) is made by weighted sum of activity, where the weighs are given by w.

n_cell = 100;
n_trial = 500;
w = randn(n_cell,1);
r1 = -1.*ones(n_trial,n_cell,1) + randn(n_trial,n_cell); % activity for stim 1 
r2 =  1.*ones(n_trial,n_cell,1) + randn(n_trial,n_cell); % activity for stim 2
s1 = 1.*ones(n_trial,n_cell,1); % stimulus labels for trials
s2 = 2.*ones(n_trial,n_cell,1); % stimulus labels for trials
c1 = r1*w>0; % choice for stim 1 trials
c2 = r2*w>0; % choice for stim 2 trials

X = [r1;r2]; % matrix of predictors -- this is called a design matrix
y = [c1;c2]; % vector of response -- choice in this example

frac_train = 0.8; % fraction of training set
pick_train = false(n_trial,1);
n_train = round(n_trial.*frac_train);
pick_train(1:n_train) = true;
pick_train = repmat(pick_train,2,1);

% split the data into train / test set
X_train = X(pick_train,:);
y_train = y(pick_train);
X_test = X(~pick_train,:);
y_test = y(~pick_train);

% use Glmnet for logistic regression
% train the GLM on the Train set
opts = struct('alpha', 0.95, 'lambda_min', exp(-6),'thresh',1E-6,'offset', [], 'intr', 1, 'standardize', 0);
options = glmnetSet(opts);
distr = 'binomial';
n_fold = 10;
cvobj = cvglmnet(X_train, y_train, distr, options, 'deviance', n_fold, []);

ind_1se = find(cvobj.lambda==cvobj.lambda_1se);
ind_min = find(cvobj.lambda==cvobj.lambda_min);

figure(1)
cvglmnetPlot(cvobj);

% pick the lambda value that gives the most regularized model such 
% that error is within one standard error of the minimum.
lambda_1se = cvobj.lambda_1se;
dev_1se = cvobj.glmnet_fit.dev(ind_1se);
a0_1se = cvobj.glmnet_fit.a0(ind_1se);
beta_1se = cvobj.glmnet_fit.beta(:,ind_1se);
df_1se = cvobj.glmnet_fit.df(ind_1se);

% Make prediction for the Test set
Xb = X_test*beta_1se + a0_1se;

% Xb is the "predicted" weighted sum of activity
% You probably want to measure the stimulation effect on Xb

c_pred = Xb>0;

mean(double(c_pred)==double(y_test))

figure(2);
plot(w,beta_1se,'ko')
