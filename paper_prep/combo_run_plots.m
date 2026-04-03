%% Plot controller error comparisons (3x2, column-wise, version-safe)
clc; close all;

% =======================
% INPUT DATA
% =======================
outsLeft  = {out1, out2, out3, out4};
outsRight = {out5, out6, out7, out8};

groups = {outsLeft, outsRight};
nGroups = 2;
nCtrl   = numel(outsLeft);

% =======================
% STYLING
% =======================
lineStyles = {'-.', ':', '--', '-'};
colors = {
    [0 0 0]
    [0 0 0]
    [0 0 0]
    [0.4 0.4 0.4]
};
lineWidth = 2;

legendLabels = {'P','PI','Fzy PID','Fzy (Prop.)'};

yLabels = {'Depth (m)','Roll Error (rad)','Yaw Error (rad)'};
cols    = [9 8 10];   % depth, roll, yaw columns

% =======================
% FIGURE & LAYOUT
% =======================
figure('Units','inches','Position',[1 1 10 5],'Color','w');
t = tiledlayout(3,2,'Padding','compact','TileSpacing','compact');

% Column-wise tile indices

% Correct column-wise tile indices for a 3x2 layout
tileMap = {
    [1 3 5];  % left column (top → bottom)
    [2 4 6];  % right column (top → bottom)
};


% =======================
% PLOTTING (COLUMN-FILL)
% =======================
for g = 1:2                      % 1 = left column, 2 = right column
    outs = groups{g};

    % Extract data
    for k = 1:nCtrl
        time{k} = squeeze(outs{k}(:,1,:));
        for r = 1:3
            err{r,k} = squeeze(outs{k}(:,cols(r),:));
        end
    end

    % Place tiles TOP → DOWN per column
    for r = 1:3

        ax = nexttile(tileMap{g}(r));
        hold(ax,'on')

        if g == 1 && r == 1
            h = gobjects(1, nCtrl);
        end
        
        for k = 1:nCtrl
            if g == 1 && r == 1
                h(k) = plot(ax, time{k}, err{r,k}, ...
                    lineStyles{k}, ...
                    'Color', colors{k}, ...
                    'LineWidth', lineWidth);
            else
                plot(ax, time{k}, err{r,k}, ...
                    lineStyles{k}, ...
                    'Color', colors{k}, ...
                    'LineWidth', lineWidth);
            end
        end
        
        ylabel(ax, yLabels{r}, 'FontSize', 12)
        set(ax,'FontSize',12)
    end
end

lgd = legend(h, legendLabels, ...
    'Orientation','horizontal', ...
    'NumColumns', nCtrl, ...
    'FontSize', 12);

lgd.Layout.Tile = 'north';   % 💡 centers legend across entire layout


xlabel(t,'Time (s)','FontSize',12)