clear; clc;
rng(123, 'twister');

% === Parameters ===
% The rate for the polynomial term in the variance of the correction term
h_r = 1/2;
% The scale of the correction term's standard deviation
cor_size = 0.5;
% The number of posterior samples to draw for Bayesian inference
N_post = 5000;
% The significance level for constructing credible intervals
alpha = 0.05;

global likLogis bd ps_bd q coef

% === Directory Setup ===
parentDir = fileparts(pwd);
addpath(pwd)
addpath(genpath(fullfile(parentDir, 'Matlab_func')))
addpath(fullfile(parentDir, 'Data'))

% === Load Data from Python ===
% The data has been pre-processed and trimmed in Python.
fname = sprintf('Real_ps_est.mat');
load(fname)
Y_data = Y;
D_data = D;
Z_data = Z;
ps_est_0 = ps_est;
% The coefficients from the propensity score model.
coef = ps_coef;
% The number of covariates (columns in Z).
[~, q] = size(Z_data);

% === GPML Setting and Function Definitions ===
likLogis = @(t) 1./(1+exp(-t));
% Bounding function for propensity scores to ensure numerical stability.
bd = @(t,ps_bd) min(max(t,ps_bd),1-ps_bd);
% Boundary parameters for trimming propensity scores.
ps_bd = 0.1;

% === Data Type Conversion and Preparation ===
% The number of effective observations after trimming
n_eff = length(Y_data);
% Converting of the data to double precision for compatibility with GPML.
Y = double(Y_data);
D = double(D_data);
Z = double(Z_data);
Ps_est = double(ps_est_0);

% === Main Part: GP Regression and ATE Estimation ===
% The Riesz representer acts as a weighting term to correct for misspecification.
Riesz_est = D./Ps_est - (1-D)./(1-Ps_est);
X = [Z, D];
% Test cases for D=0 (control group).
X_t0 = [Z, zeros(n_eff,1)];
% Test cases for D=1 (treated group).
X_t1 = [Z, ones(n_eff,1)];
X_t = [X_t0; X_t1];

% The GP regression parameters.
inf_method = @infExact;
meanfunc = @meanConst; hyp.mean = mean(Y);
covfunc = @covSEard;
% The number of parameters is the number of covariates
% + 1 for the treatment variable + 1 for the scale.
hyp.cov = log([ones(1,q+1), std(Y)^2]);
% Likelihood function for a continuous outcome (Y).
likfunc = @likGauss;
% Hyperparameter for the likelihood function (noise variance).
hyp.lik = log(std(Y));

% === 1. Standard Bayesian Inference (Without Correction) ===
% Minimization of the negative log marginal likelihood to find optimal hyperparameters.
hyp = minimize(hyp, @gp, -30, @infExact, meanfunc, covfunc, likfunc, X, Y);
% Posterior inference to get the mean and covariance of the GP.
[m_f_reg_mu_nc, m_f_reg_cov_nc] = gp(hyp, inf_method, meanfunc, covfunc, likfunc, X, Y, X_t);

% Stabilize the covariance matrix to ensure it is symmetric positive definite.
try
    chol(m_f_reg_cov_nc);
catch ME
    m_f_reg_cov_nc = (m_f_reg_cov_nc + m_f_reg_cov_nc')/2;
    [ev_0,ed_0] = eig(m_f_reg_cov_nc);
    ed_p0 = max(diag(ed_0),1.0e-6);
    m_f_reg_cov_nc = ev_0*diag(ed_p0)/ev_0;
end
% Drawing posterior samples of the GP function from the posterior distribution.
M_f_reg_nc = repmat(m_f_reg_mu_nc,1,N_post) + chol(m_f_reg_cov_nc,'lower')*randn(2*n_eff,N_post);
M_f_reg_nc = M_f_reg_nc';
% Samples for the control group (D=0) and treated group (D=1).
M_f_nc_0 = M_f_reg_nc(:,1:n_eff);
M_f_nc_1 = M_f_reg_nc(:,n_eff+1:2*n_eff);
% Samples for the observed outcomes.
M_f_nc_obs = repmat(D',N_post,1).*M_f_reg_nc(:,n_eff+1:2*n_eff) + repmat((1-D)',N_post,1).*M_f_reg_nc(:,1:n_eff);

% === 2. Bayesian Inference with Prior Correction (PA Bayes) ===
% The standard deviation of the correction term.
cor_sd_est =  cor_size * log(n_eff)/((n_eff)^(h_r)*mean(abs(Riesz_est)));
% `covPSlogi` uses the PS
% to add a correction term to the prior.
cov_cor_hat =  {'covSum',{'covSEard','covPSlogi'}};
% Hyperparameters for the corrected model.
hyp_cor.mean = hyp.mean;
% Additing the new hyperparameter for the correction term.
hyp_cor.cov = [hyp.cov, log(cor_sd_est)];
hyp_cor.lik = hyp.lik;
% Posterior inference for the corrected model.
[m_f_reg_mu, m_f_reg_cov] = gp(hyp_cor, inf_method, meanfunc, cov_cor_hat, likfunc, X, Y, X_t);

% Stabilize the covariance matrix for the corrected model.
try
    chol(m_f_reg_cov);
catch ME
    m_f_reg_cov = (m_f_reg_cov + m_f_reg_cov')/2;
    [ev,ed] = eig(m_f_reg_cov);
    ed_p = max(diag(ed),1.0e-6);
    m_f_reg_cov = ev*diag(ed_p)/ev;
end
% Drawing posterior samples from the corrected model.
M_f_reg = repmat(m_f_reg_mu,1,N_post) + chol(m_f_reg_cov,'lower')*randn(2*n_eff,N_post);
M_f_reg = M_f_reg';
% Samples for the control group (D=0) and treated group (D=1).
M_f_0 = M_f_reg(:,1:n_eff);
M_f_1 = M_f_reg(:,n_eff+1:2*n_eff);
% Samples for the observed outcomes.
M_f_obs = repmat(D',N_post,1).*M_f_reg(:,n_eff+1:2*n_eff) + repmat((1-D)',N_post,1).*M_f_reg(:,1:n_eff);

% === Bayesian Bootstrap ===
% Drawing weights from a Dirichlet process for the Bayesian bootstrap.
DP_weights_0 = exprnd(1,[N_post,n_eff]);
DP_weights = DP_weights_0./repmat(sum(DP_weights_0,2),1,n_eff);

% (1) Estimation of ATE with the standard Bayesian method.
Ate = sum((M_f_nc_1-M_f_nc_0).*DP_weights,2);
Ate_m = mean(Ate);
Ate_low = quantile(Ate,alpha/2); Ate_up = quantile(Ate,1-alpha/2);
bayes_std = std(Ate);

% (2) Estimation of ATE with the PA Bayes (Prior Correction) method.
Ate_pc = sum((M_f_1-M_f_0).*DP_weights,2);
Ate_pc_m = mean(Ate_pc);
Ate_pc_low = quantile(Ate_pc,alpha/2);Ate_pc_up = quantile(Ate_pc,1-alpha/2);
pa_bayes_std = std(Ate_pc);

% (3) Estimation of ATE with the Double Robust (DR) Bayes method.
% This method combines the model-based estimate and the Riesz representer
% to ensure robustness.
ATE_dr_pre = mean(sum((M_f_nc_1-M_f_nc_0),1)/N_post + Riesz_est'.*(Y'-sum(M_f_nc_obs,1)/N_post));
DR_rec_1 = repmat(Riesz_est',N_post,1).*(repmat(Y',N_post,1)-M_f_obs);
Ate_drb = sum((M_f_1-M_f_0).*DP_weights,2) + ATE_dr_pre - sum(DR_rec_1/n_eff,2) - sum((M_f_1-M_f_0)/n_eff,2);
Ate_drb_m = mean(Ate_drb);
Ate_drb_low = quantile(Ate_drb,alpha/2);Ate_drb_up = quantile(Ate_drb,1-alpha/2);
dr_bayes_std = std(Ate_drb);

% === Results ===
disp('------------------------------------------------------------')
disp('Method         Mean      Std      95% CI')
disp('------------------------------------------------------------')
fprintf('Bayes:       %.3f   %.3f   [%.3f, %.3f]\n', Ate_m, bayes_std, Ate_low, Ate_up);
fprintf('PA Bayes:    %.3f   %.3f   [%.3f, %.3f]\n', Ate_pc_m, pa_bayes_std, Ate_pc_low, Ate_pc_up);
fprintf('DR Bayes:    %.3f   %.3f   [%.3f, %.3f]\n', Ate_drb_m, dr_bayes_std, Ate_drb_low, Ate_drb_up);
disp('------------------------------------------------------------')

% === Save results to a file for Python ===
% Direction to `master_thesis`
master_thesis_dir = fullfile(fileparts(fileparts(pwd))); 

% Direction to `data/obtained_data`
results_dir = fullfile(master_thesis_dir, 'data', 'obtained_data');

fname_out = fullfile(results_dir, 'Bayesian_results.txt');

% Writing results to file
fileID = fopen(fname_out, 'w');
fprintf(fileID, 'bayes_mean=%.3f\n', Ate_m);
fprintf(fileID, 'bayes_std=%.3f\n', bayes_std);
fprintf(fileID, 'pa_bayes_mean=%.3f\n', Ate_pc_m);
fprintf(fileID, 'pa_bayes_std=%.3f\n', pa_bayes_std);
fprintf(fileID, 'dr_bayes_mean=%.3f\n', Ate_drb_m);
fprintf(fileID, 'dr_bayes_std=%.3f\n', dr_bayes_std);
fclose(fileID);

disp(['Results saved to ' fname_out]);