% Generate the data, RMSE table, and plots corresponding to Figure 11.
project_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(project_dir, 'paper_preperation'));

[figure11_runs, figure11_rmse, figure11_figure] = ...
    controller_comarision_results();
