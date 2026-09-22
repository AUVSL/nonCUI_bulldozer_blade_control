close all; clc;
script_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(script_dir);
controllers_dir = fullfile(project_dir, 'controllers');
output_dir = fullfile(script_dir, 'generated_controller_plots');
addpath(genpath(controllers_dir));
if ~isfolder(output_dir)
    mkdir(output_dir);
end
soil = 0.1;
run(fullfile(project_dir, 'parameters.m'));

fis1 = readfis('paper_blade_ang.fis');
fis1_p = set_angle_output_gain(fis1, KpProp);
fis1_i = set_angle_output_gain(fis1, KiProp);
fis2 = readfis('blade_height.fis');
% plotfis(fis)

plotter2(fis1_p, 'input', 1, [0 0.4 0 1], ...
    {'zero', 'non-zero'}, 2, {'-', '--'}, ...
    fullfile(output_dir, 'blade_angle_input.svg'))
plotter2(fis1_p, 'output', 1, output_axis(KpProp), ...
    {'zero', 'non-zero'}, 2, {'-', '--'}, ...
    fullfile(output_dir, 'blade_angle_p_output.svg'))
plotter2(fis1_i, 'output', 1, output_axis(KiProp), ...
    {'zero', 'non-zero'}, 2, {'-', '--'}, ...
    fullfile(output_dir, 'blade_angle_i_output.svg'))

plotter2(fis2, 'input', 1, [0 1 0 1], ...
    {'compact', 'mixed', 'loose'}, 3, {'--', '-', '-.'}, ...
    fullfile(output_dir, 'blade_height_input.svg'))
plotter2(fis2, 'output', 1, [-0.041 -0.029 0 1], ...
    {'low', 'lower', 'lowest'}, 3, {'--', '-', '-.'}, ...
    fullfile(output_dir, 'blade_height_output.svg'))

fprintf('Saved controller membership plots to: %s\n', output_dir);

function fis = set_angle_output_gain(fis, gain_value)
    half_width = 0.5;
    fis.Outputs(1).Range = [gain_value - half_width, half_width];
    fis.Outputs(1).MembershipFunctions(1).Parameters = ...
        [-half_width, 0, half_width];
    fis.Outputs(1).MembershipFunctions(2).Parameters = ...
        gain_value + [-half_width, 0, half_width];
end

function limits = output_axis(gain_value)
    limits = [floor(gain_value - 0.5), 1, 0, 1];
end

function[] = plotter2(fis, io, x1, x2, x3, x4, x5, svg_file)
    f = figure;
    f.Position = [644,665,597,313];
    [x,mf] = plotmf(fis, io, x1);
    subplot(1,1,1)
    p = plot(x, mf, 'LineWidth',2);
    axis(x2)
    legend(x3, 'Location','northoutside', 'Orientation', 'horizontal', 'FontSize', 14, 'NumColumns', x4)
    set(gca,"FontSize", 14)
    NameArray = {'LineStyle'};
    ValueArray = transpose(x5);
    set(p,NameArray,ValueArray)
    set(p, 'Color', [0, 0, 0])
    saveas(f, svg_file)
end
