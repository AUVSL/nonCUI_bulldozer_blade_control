close all; clc;
script_dir = fileparts(mfilename('fullpath'));
controllers_dir = fullfile(fileparts(script_dir), 'controllers');
addpath(genpath(controllers_dir));
fis1 = readfis('blade_ang.fis');
fis2 = readfis('blade_height.fis');
% plotfis(fis)

plotter2(fis1,  'input', 1, [0 2e-2 0 1], {'zero', 'non-zero'}, 3, {'--', '-'})
plotter2(fis1, 'output', 1, [-0.2 1.2 0 1], {'zero', 'non-zero'}, 3, {'--', '-'})

plotter2(fis2,  'input', 1, [0 1. 0 1], {'compact', 'loose'}, 3, {'--', '-'})
plotter2(fis2, 'output', 1, [-2e-3 -5e-4 0 1], {'low', 'lower'}, 3, {'--', '-'})

function[] = plotter2(fis, io, x1, x2, x3, x4, x5)
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
end
