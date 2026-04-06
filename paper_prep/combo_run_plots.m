%% Plot controller error comparisons (3x3, column-wise, version-safe)
clc; close all;

% =======================
% INPUT DATA
% =======================
outsLeft  = {out1, out2, out3, out4};
outsMid   = {out9, out10, out11, out12};
outsRight = {out5, out6, out7, out8};

groups  = {outsLeft, outsMid, outsRight};
nGroups = numel(groups);
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

yLabels = {'Dp Er (m)','Rl Er (rad)','Yw Er (rad)'};
cols    = [9 8 10];   % depth, roll, yaw columns

% =======================
% FIGURE & LAYOUT
% =======================
figure('Units','inches','Position',[1 1 10 5],'Color','w');
t = tiledlayout(3,3,'Padding','compact','TileSpacing','compact');

% Column-wise tile indices (TOP → BOTTOM)
tileMap = {
    [1 4 7];   % left column
    [2 5 8];   % middle column
    [3 6 9];   % right column
};

% =======================
% PLOTTING (COLUMN-FILL)
% =======================
for g = 1:nGroups
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

        % Capture legend handles once
        if g == 1 && r == 1
            h = gobjects(1,nCtrl);
        end

        for k = 1:nCtrl
            if g == 1 && r == 1
                h(k) = plot(ax,time{k},err{r,k}, ...
                    lineStyles{k},'Color',colors{k}, ...
                    'LineWidth',lineWidth);
            else
                plot(ax,time{k},err{r,k}, ...
                    lineStyles{k},'Color',colors{k}, ...
                    'LineWidth',lineWidth);
            end
        end

        if g == 1
            ylabel(ax,yLabels{r},'FontSize',12)   % first column only
        else
            ax.YTickLabel = [];                  % remove y‑tick text
        end

        set(ax,'FontSize',12)
    end
end

% =======================
% LEGEND & LABELS
% =======================
lgd = legend(h,legendLabels, ...
    'Orientation','horizontal', ...
    'NumColumns',nCtrl, ...
    'FontSize',14);

lgd.Layout.Tile = 'north';   % centered across entire layout
xlabel(t,'Time (s)','FontSize',12)
saveas(gcf,'paper_prep/results.svg')