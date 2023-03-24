clc; close all; clear

sfi = 0.05;
clim = [0.2001,1];
% clim = [0.59, 1];
mu_max = 0.5;     %% maximum mu

row_cmap = 150;  
mycolormap=zeros(row_cmap,3);  

% % blue
color_r = 0:1/(row_cmap-1):1; 
color_g = 0:1/(row_cmap-1):1;
color_b = ones(1,row_cmap);

% % red
% color_r = ones(1,row_cmap);
% color_g = 0:1/(row_cmap-1):1;
% color_b =  0:1/(row_cmap-1):1;

mycolormap(:,1) = color_r; 
mycolormap(:,2) = color_g;
mycolormap(:,3) = color_b;

colorbar;


%%  plots

r = 0:0.05:1.0;  
mu = 0.05:0.05:mu_max;

figure(1)  % fig4： MuR_KNS
load('./Data_MuR_KNS.mat')
min(min(Data_MuR_KNS))

h = imagesc(r,mu,Data_MuR_KNS,clim);
set(h,'alphadata',~isnan(Data_MuR_KNS))

c=colorbar;
fs = 13;
set(c,'YTick',0.3:0.1:1); %色标值范围及显示间隔
set(c,'YTickLabel',{'0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'}) %具体刻度赋值
set(c,'LineWidth',1,'FontSize',fs)
colormap(mycolormap)
axis xy 
set(gca,'LineWidth',1.,'FontSize',fs);
figure_FontSize = 20;
ylabel('\mu','fontsize',figure_FontSize);
xlabel('{\itr}','FontSize',figure_FontSize, ...
       'FontName', 'Times New Roman');
xticks([0 0.2 0.4 0.6 0.8 1])
% mus
ylim([0.05 0.5])
yticks(0.05:0.05:0.50)

%%

figure(2)  % fig4： MuR_GA
load('./Data_MuR_GA.mat')
min(min(Data_MuR_GA))

h1 = imagesc(r,mu,Data_MuR_GA,clim);
set(h1,'alphadata',~isnan(Data_MuR_GA))

c=colorbar;
fs = 13;
set(c,'YTick',0.3:0.1:1); %色标值范围及显示间隔
set(c,'YTickLabel',{'0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'}) %具体刻度赋值
set(c,'LineWidth',1,'FontSize',fs)
colormap(mycolormap)
axis xy 
set(gca,'LineWidth',1.,'FontSize',fs);
figure_FontSize = 20;
ylabel('\mu','fontsize',figure_FontSize);
xlabel('{\itr}','FontSize',figure_FontSize, ...
       'FontName', 'Times New Roman');
xticks([0 0.2 0.4 0.6 0.8 1])
% mus
ylim([0.05 0.5])
yticks(0.05:0.05:0.5)


%%

r = 0:0.05:1.0;  
sigma_fij = 0.005:0.005:0.05;

figure(3)  % Fig4: SigmaR_KNS
load('./Data_SigmaR_KNS.mat')
min(min(Data_SigmaR_KNS))

h2 = imagesc(r,sigma_fij,Data_SigmaR_KNS,clim);
set(h2,'alphadata',~isnan(Data_SigmaR_KNS))

c=colorbar;
fs = 13;
set(c,'YTick',0.3:0.1:1); %色标值范围及显示间隔
set(c,'YTickLabel',{'0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'}) %具体刻度赋值
set(c,'LineWidth',1,'FontSize',fs)
colormap(mycolormap)
axis xy 
set(gca,'LineWidth',1.,'FontSize',fs);
figure_FontSize = 20;
xlabel('{\itr}','FontSize',figure_FontSize, ...
       'FontName', 'Times New Roman');
xticks([0 0.2 0.4 0.6 0.8 1])
ylabel('\sigma_{e}','FontSize',figure_FontSize); % , 'fontweight','bold'
ylim([0.004 0.05])
yticks([0.005 0.01 0.015  0.02 0.025  0.03 0.035 0.04 0.045 0.05])

%%
figure(4)  % Fig4: SigmaR_GA
load('./Data_SigmaR_GA.mat')
min(min(Data_SigmaR_GA))

h3 = imagesc(r,sigma_fij,Data_SigmaR_GA,clim);
set(h3,'alphadata',~isnan(Data_SigmaR_GA))

c=colorbar;
fs = 13;
set(c,'YTick',0.3:0.1:1); %色标值范围及显示间隔
set(c,'YTickLabel',{'0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'}) %具体刻度赋值
set(c,'LineWidth',1,'FontSize',fs)
colormap(mycolormap)
axis xy 
set(gca,'LineWidth',1.,'FontSize',fs);
figure_FontSize = 20;
xlabel('{\itr}','FontSize',figure_FontSize, ...
       'FontName', 'Times New Roman');
xticks([0 0.2 0.4 0.6 0.8 1])
ylabel('\sigma_{e}','FontSize',figure_FontSize); % , 'fontweight','bold'
ylim([0.004 0.05])
yticks([0.005 0.01 0.015  0.02 0.025  0.03 0.035 0.04 0.045 0.05])


