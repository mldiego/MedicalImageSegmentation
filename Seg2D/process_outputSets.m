%% Process results for all instances verified for one model

model = [64];
% models = [64, 80, 96, 112, 128];

% adversarial perturbation: linf
results = dir("results/*linf_"+string(model)+"*.mat");

% compute riou and % rob for every reach set computed

% do we want to verify anything else?

