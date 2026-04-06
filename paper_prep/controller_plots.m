close all; clc; clear
addpath(genpath('C:\Users\srd2\Code\nonCUI_bulldozer_blade_control\controllers'));
fis1 = readfis('paper_blade_ang.fis');
fis2 = readfis('blade_height.fis');
% plotfis(fis)

plotter2(fis1,'input',1,[0 0.4 0 1],{'zero','non-zero'},3,{'--','-'}, ...
         'paper_prep/in_ang.svg')

plotter2(fis1,'output',1,[-16 1 0 1],{'zero','non-zero'},3,{'--','-'}, ...
         'paper_prep/out_ang.svg')

plotter2(fis2,'input',1,[0 1 0 1],{'compact','loose'},3,{'--','-'}, ...
         'paper_prep/in_height.svg')

plotter2(fis2,'output',1,[-4.5e-2 -2.5e-2 0 1],{'low','lower'},3,{'--','-'}, ...
         'paper_prep/out_height.svg')

function plotter2(fis, io, x1, x2, x3, x4, x5, filename)

    f = figure;
    f.Position = [644,665,597,313];

    [x,mf] = plotmf(fis, io, x1);
    p = plot(x, mf, 'LineWidth', 2);
    axis(x2)

    legend(x3, ...
        'Location','northoutside', ...
        'Orientation','horizontal', ...
        'FontSize',14, ...
        'NumColumns',x4)

    set(gca,'FontSize',14)

    % ✅ Assign styles safely (one per line)
    for k = 1:numel(p)
        p(k).LineStyle = x5{k};
        p(k).Color     = [0 0 0];
    end

    % ✅ Save as SVG
    saveas(f, filename)

end