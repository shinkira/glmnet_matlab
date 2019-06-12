function GlmForNoah
    
    rng(0);

    lambda = 10*rand(1,8); % expected spike counts for 8 trial types
    
    x_txt = {'B','B','W','W','B','B','W','W';
             'WB','BW','BW','WB','WB','BW','BW','WB';
             'R','L','R','L','L','R','L','R',};
              
    figure(1)
    % plot(0:8,0:8,'k--');
    bar(1:8,lambda);
    set(gca,'XTick',1:8,'XTickLabel',[]);
    for i = 1:8
        text(i,-1/3,x_txt{1,i});
        text(i,-2/3,x_txt{2,i});
        text(i,-3/3,x_txt{3,i});
    end

    dm = [-1  -1   1   1  -1  -1   1   1; % sample
          -1   1   1  -1  -1   1   1  -1; % test
           1  -1   1  -1  -1   1  -1   1]'; % choice
       
    dm2 = randn(8,20);
    % dm2 = double(randn(8,100)>0);
    % dm2(dm2==0) = -1;
        
    n_trial = 200;
    n_trial_cuePos = n_trial/4;
    p_cor = 0.8;
    
    correct_v = false(n_trial_cuePos,1);
    correct_v(1:n_trial_cuePos*p_cor) = true;
    
    cuePos = [];
    correct = [];
    
    for gi = 1:4
        cuePos = [cuePos;gi*ones(n_trial_cuePos,1)];
        correct = [correct;correct_v];
    end
    
    cuePos8 = round(cuePos + 4*(1-correct))';
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                  Cross validation parameters                        %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    n_split = 5; % number of train/test repeats (use 1 for debugging)
    frac_train = 0.8; % fraction of training trais among all valid trials
    n_fold = 10; % number of cross-validation folds within trainig trials
    
    split_ind = [];
    trial_ind_all = [];
    cuePosSorted = [];
    for gi = 1:8 % loop through cuePos8
        trial_ind_cuePos = find(cuePos8==gi);
        if ~isempty(trial_ind_cuePos)
            n_trial_cuePos8 = length(trial_ind_cuePos);
            split_ind = [split_ind,crossvalind('kfold',n_trial_cuePos8,n_split)'];
            trial_ind_all = [trial_ind_all,trial_ind_cuePos];
            cuePosSorted = [cuePosSorted,gi*ones(1,n_trial_cuePos8)];
        end
    end

    for si = 1:n_split
        pick_test = split_ind==si;
        trial_ind_train{si} = trial_ind_all(~pick_test);
        trial_ind_test{si}  = trial_ind_all(pick_test);
        cuePosSorted_train{si} = cuePosSorted(~pick_test);
    end

    for si = 1:n_split
        fold_ind{si} = [];
        n_folds(si) = n_fold;
        for gi = 1:8
            n = sum(cuePosSorted_train{si}==gi);
            if n>0
                fold_ind{si} = [fold_ind{si},crossvalind('kfold', n, n_fold)'];
            end
        end
    end
    
    for gi = 1:8
        pick = cuePos8==gi;
        y(pick,1) = poissrnd(lambda(gi),sum(pick),1);
    end
    
    opts = struct('alpha', 0.95, 'lambda_min', exp(-6),'thresh',1E-6,'offset', [], 'intr', 1, 'standardize', 0);
    options = glmnetSet(opts);
    distr = 'poisson';
    n_fold = 10;
    
    X = dm(cuePos8,:);
    % X = eye(n_trial);
    X = [X,eye(n_trial)];
    
    X_train = X(trial_ind_train{1},:);
    y_train = y(trial_ind_train{1});
    X_test = X(trial_ind_test{1},:);
    y_test = y(trial_ind_test{1});
    
    %% GLM is trained on the training set and evaluated on the TRAINING set
    options.data2eval = 'train';
    cvobj = cvglmnet(X_train, y_train, distr, options, 'deviance', n_fold, fold_ind{1});
    figure(2);clf;
    cvglmnetPlot(cvobj);
    set(gcf,'OuterPosition',[50 500 1000 650])
    y_lim = get(gca,'Ylim');
    
    %% GLM is trained on the training set and evaluated on the TEST set
    options.data2eval = 'test';
    cvobj = cvglmnet(X_train, y_train, distr, options, 'deviance', n_fold, fold_ind{1});
    figure(3);clf;
    cvglmnetPlot(cvobj);
    set(gcf,'OuterPosition',[1050 500 1000 650])
    ylim(y_lim);
    
    %%
    % evaluate @ lombda 1se away from the lambda_min
    lambda_1se = cvobj.lambda_1se;
    ind_1se = find(cvobj.lambda==cvobj.lambda_1se);
    dev_1se = cvobj.glmnet_fit.dev(ind_1se);
    a0_1se = cvobj.glmnet_fit.a0(ind_1se);
    beta_1se = cvobj.glmnet_fit.beta(:,ind_1se);
    
    % evaluate @ lambda_min
    lambda_min = cvobj.lambda_min;
    ind_min = find(cvobj.lambda==cvobj.lambda_min);
    dev_min = cvobj.glmnet_fit.dev(ind_min);
    a0_min = cvobj.glmnet_fit.a0(ind_min);
    beta_min = cvobj.glmnet_fit.beta(:,ind_min);
    
    % evauate @ lambda very close to zero (i.e., no regularization)
    lambda_zero = cvobj.lambda(end);
    ind_zero = length(cvobj.lambda);
    dev_zero = cvobj.glmnet_fit.dev(ind_zero);
    a0_zero = cvobj.glmnet_fit.a0(ind_zero);
    beta_zero = cvobj.glmnet_fit.beta(:,ind_zero);
    
    % evaluate without cross validation
    [beta_nocv,dev_nocv,stats] = glmfit(X_train,y_train,'poisson');
    a0_nocv = beta_nocv(1);
    beta_nocv = beta_nocv(2:end);
    
    Xb_1se   = X_test*beta_1se  + a0_1se;
    Xb_min   = X_test*beta_min  + a0_min;
    Xb_zero  = X_test*beta_zero + a0_zero;
    Xb_nocv  = X_test*beta_nocv + a0_nocv;
    Xb_null = log(mean(y_train)).*ones(size(y_test)); % the null model predicts a constant activity
    
    dev_1se_test = mean(devi(y_test, Xb_1se));
    dev_min_test = mean(devi(y_test, Xb_min));
    dev_zero_test = mean(devi(y_test, Xb_zero));
    dev_nocv_test = mean(devi(y_test, Xb_nocv));
    dev_null_test = mean(devi(y_test, Xb_null));
    
    % fraction of deviance explained
    fde_1se = (dev_null_test - dev_1se_test)/dev_null_test;
    fde_min = (dev_null_test - dev_min_test)/dev_null_test;
    fde_zero = (dev_null_test - dev_zero_test)/dev_null_test;
    fde_nocv = (dev_null_test - dev_nocv_test)/dev_null_test;
    
end

function result = devi(yy, eta)
    % copied from cvfishnet.m
    deveta = yy .* eta - exp(eta); % fitted model
    devy = yy .* log(yy) - yy; % saturated model
    devy(yy == 0) = 0;
    result = 2 * (devy - deveta);
end
