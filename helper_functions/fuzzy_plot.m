close all; clc;
addpath(genpath('C:\Users\srd2\Code\nonCUI_bulldozer_blade_control\controllers'));
fis1 = readfis('blade_ang.fis');
fis2 = readfis('blade_height.fis');
% plotfis(fis)

plotter2(fis1,  'input', 1, [-3.2 3.2 0 1], {'neg', 'zero', 'pos'}, 3, {'--', '-', ':'})
plotter2(fis1, 'output', 1, [-1.2 1.2 0 1], {'neg', 'zero', 'pos'}, 3, {'--', '-', ':'})

plotter2(fis2,  'input', 1, [0 1. 0 1], {'compact', 'mixed', 'loose'}, 3, {'--', '-', ':'})
plotter2(fis2, 'output', 1, [-0.35 -0.15 0 1], {'low', 'lower', 'lowest'}, 3, {'--', '-', ':'})

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