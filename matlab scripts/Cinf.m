function Cinf
% computation of an invariant set
%
% M. Herceg, Automatic Control Laboratory, ETH Zurich 2014
%
% requires MPT3

close all

%% printing parameters
label_font_size = 14;
tick_font_size = 10;
line_width = 0.8;
axeswidth=0.2;
figure_name = ['.',filesep,'figures',filesep,'invariantSet_ex_12_4'];
iF = 1;
to_write = false;
to_plot = true;

%% ---------------- Example A -----------------
% x(k+1) = A*x(k) + B*u(k)
A = [1 1 ; 0 1];
B = [0 ; 1];

% create model in MPT3 interface
model = LTISystem('A',A,'B',B);

% constraints on inputs and states
% model.u.min = -5;
% model.u.max = 5;
% model.x.min = [-10;-10];
% model.x.max = [ 10; 10];

model.x.min = [-5;-5];
model.x.max = [ 5; 5];
model.u.min = -1;
model.u.max = 1;


% constraint sets represented as polyhedra
X = Polyhedron('lb',model.x.min,'ub',model.x.max);
U = Polyhedron('lb',model.u.min,'ub',model.u.max);

% compute the invariant set including the intermediate steps
maxIterations = 100;
Piter = []; % save intermediate set during computation
X0 = X; % initial set constraint
for i = 1:maxIterations
    % compute backward reachable set
    P = model.reachableSet('X', X0, 'U', U, ...
        'direction', 'backward');
    % intersect with the state constraints
    P = P.intersect(X0).minHRep();
    Piter = [Piter, P];
    if P==X
        break
    else
        X0 = P;
    end
end


% direct command for computing the invariant set (but here we want to show
% intermediate sets)
% Cinf = model.invariantSet(); 

% the invariant set
Cinf = P;
CinfH = Cinf.H;
CinfV = Cinf.V;

if to_plot
    figure(1)
    % plot the constraint set
    plot(X,'color',[0.7 0.7 0.7],'linewidth',line_width);
    hold on
    grid on
    % plot the intermediate sets
    hp = plot(Piter,'color',[0.9 0.9 0.9]);
    % plot the invariant set
    plot(Cinf,'color',[0.7 0.7 0.7],'linewidth',line_width)
    axis([model.x.min(1),model.x.max(1),model.x.min(2),model.x.max(2)])
    %title('Cinf')

    set(gca,'LineWidth',axeswidth)
    set(gca,'FontSize', tick_font_size);
    xt = transpose(-10:5:10);
    yt = transpose(-10:5:10);
    set(gca,'XTick',xt);
    set(gca,'YTick',yt);
    set(gca,'YTickLabel',num2str(xt));
    set(gca,'XTickLabel',num2str(yt));

    hx1 = xlabel('$x_1$', 'interpreter', 'latex');
    set(hx1, 'FontSize', label_font_size);
    hy1 = ylabel('$x_2$', 'interpreter', 'latex');
    set(hy1, 'FontSize', label_font_size);

    ht1=text(7,-8,'$\mathcal{X}$', 'interpreter', 'latex');
    set(ht1, 'FontSize', label_font_size);
    ht2=text(-1,-0.5,'$\mathcal{C}_{\infty}$', 'interpreter', 'latex');
    set(ht2, 'FontSize', label_font_size);

    border = 1;
    axis([model.x.min(1)-border,model.x.max(1)+border,model.x.min(2)-border,model.x.max(2)+border])

    figure(2)
    % plot the constraint set
    plot(X,'color',[0.7 0.7 0.7],'linewidth',line_width);
    %hold on
    %grid on
    % plot the intermediate sets
    %hp = plot(Piter,'color',[0.7 0.7 0.7]);
    grid on;
    % plot the invariant set
    plot(Cinf,'color',[0.7 0.7 0.7],'linewidth',line_width)
    axis([model.x.min(1),model.x.max(1),model.x.min(2),model.x.max(2)])
    %title('Cinf')

    set(gca,'LineWidth',axeswidth)
    set(gca,'FontSize', tick_font_size);
    xt = transpose(-10:5:10);
    yt = transpose(-10:5:10);
    set(gca,'XTick',xt);
    set(gca,'YTick',yt);
    set(gca,'YTickLabel',num2str(xt));
    set(gca,'XTickLabel',num2str(yt));

    hx1 = xlabel('$x_1$', 'interpreter', 'latex');
    set(hx1, 'FontSize', label_font_size);
    hy1 = ylabel('$x_2$', 'interpreter', 'latex');
    set(hy1, 'FontSize', label_font_size);

    ht1=text(7,-8,'$\mathcal{X}$', 'interpreter', 'latex');
    set(ht1, 'FontSize', label_font_size);
    ht2=text(-1,-0.5,'$\mathcal{C}_{\infty}$', 'interpreter', 'latex');
    set(ht2, 'FontSize', label_font_size);

    border = 1;
    axis([model.x.min(1)-border,model.x.max(1)+border,model.x.min(2)-border,model.x.max(2)+border])

end


% write system dynamics to file
if to_write
    dlmwrite('\ExA\exA_A.csv', A)
    dlmwrite('\ExA\exA_B.csv', B)
    dlmwrite('\ExA\exA_xlb.csv', model.x.min)
    dlmwrite('\ExA\exA_xub.csv', model.x.max)
    dlmwrite('\ExA\exA_ulb.csv', model.u.min)
    dlmwrite('\ExA\exA_uub.csv', model.u.max)
    dlmwrite('\ExA\exA_cinfH.csv', Cinf.H)
    dlmwrite('\ExA\exA_cinfV.csv', Cinf.V)
end



%% ---------------- Example B -----------------


%% ---------------- Example C -----------------


%% ---------------- Example D -----------------


% x(k+1) = A*x(k) + B*u(k)
A = [1 1 ; 0 1];
B = [0 ; 1];

% create model in MPT3 interface
model = LTISystem('A',A,'B',B);

% constraints on inputs and states
% model.u.min = -5;
% model.u.max = 5;
% model.x.min = [-10;-10];
% model.x.max = [ 10; 10];

model.x.min = [-5;-5];
model.x.max = [ 5; 5];
model.u.min = -0.5;
model.u.max = 0.5;

% write system dynamics to file
% dlmwrite('A.csv', A)
% dlmwrite('B.csv', B)
% dlmwrite('xlb.csv', model.x.min)
% dlmwrite('xub.csv', model.x.max)
% dlmwrite('ulb.csv', model.u.min)
% dlmwrite('uub.csv', model.u.max)


% constraint sets represented as polyhedra
X = Polyhedron('lb',model.x.min,'ub',model.x.max);
U = Polyhedron('lb',model.u.min,'ub',model.u.max);

% compute the invariant set including the intermediate steps
maxIterations = 100;
Piter = []; % save intermediate set during computation
X0 = X; % initial set constraint
for i = 1:maxIterations
    % compute backward reachable set
    P = model.reachableSet('X', X0, 'U', U, ...
        'direction', 'backward');
    % intersect with the state constraints
    P = P.intersect(X0).minHRep();
    Piter = [Piter, P];
    if P==X
        break
    else
        X0 = P;
    end
end


% direct command for computing the invariant set (but here we want to show
% intermediate sets)
% Cinf = model.invariantSet(); 

% the invariant set
Cinf = P;
CinfH = Cinf.H;
CinfV = Cinf.V;
%sH = size(Cinf.H);
%sV = size(CinfV);
%classCinf = class(Cinf)

plot_and_write = true;
if plot_and_write    
    figure(1)
    % plot the constraint set
    % plot(X,'color',[0.7 0.7 0.7],'linewidth',line_width);
    hold on
    grid on
    % plot the intermediate sets
    hp = plot(Piter,'color',[0.7 0.7 0.7]);
    % plot the invariant set
    plot(Cinf,'color',[0.5 0.5 0.5],'linewidth',line_width)
    axis([model.x.min(1),model.x.max(1),model.x.min(2),model.x.max(2)])


    set(gca,'LineWidth',axeswidth)
    set(gca,'FontSize', tick_font_size);
    xt = transpose(-10:5:10);
    yt = transpose(-10:5:10);
    set(gca,'XTick',xt);
    set(gca,'YTick',yt);
    set(gca,'YTickLabel',num2str(xt));
    set(gca,'XTickLabel',num2str(yt));

    hx1 = xlabel('$x_1$', 'interpreter', 'latex');
    set(hx1, 'FontSize', label_font_size);
    hy1 = ylabel('$x_2$', 'interpreter', 'latex');
    set(hy1, 'FontSize', label_font_size);

    ht1=text(7,-8,'$\mathcal{X}$', 'interpreter', 'latex');
    set(ht1, 'FontSize', label_font_size);
    ht2=text(-1,-0.5,'$\mathcal{C}_{\infty}$', 'interpreter', 'latex');
    set(ht2, 'FontSize', label_font_size);

    border = 1;
    axis([model.x.min(1)-border,model.x.max(1)+border,model.x.min(2)-border,model.x.max(2)+border])

    %figure(2)
    %plot(X,'color',[1 1 1],'linewidth',line_width);
    %hold on
    %grid on
%     plot(Cinf, 'color', [0.7 0.7 0.7], 'linewidth', line_width)
%     hold on;
%     axis([model.x.min(1),model.x.max(1),model.x.min(2),model.x.max(2)])
%     title('Cinf.H')
% 
%     set(gca,'LineWidth',axeswidth)
%     set(gca,'FontSize', tick_font_size);
%     xt = transpose(-10:5:10);
%     yt = transpose(-10:5:10);
%     set(gca,'XTick',xt);
%     set(gca,'YTick',yt);
%     set(gca,'YTickLabel',num2str(xt));
%     set(gca,'XTickLabel',num2str(yt));
% 
%     hx1 = xlabel('$x_1$', 'interpreter', 'latex');
%     set(hx1, 'FontSize', label_font_size);
%     hy1 = ylabel('$x_2$', 'interpreter', 'latex');
%     set(hy1, 'FontSize', label_font_size);
% 
%     ht1=text(7,-8,'$\mathcal{X}$', 'interpreter', 'latex');
%     set(ht1, 'FontSize', label_font_size);
%     ht2=text(-1,-0.5,'$\mathcal{C}_{\infty}$', 'interpreter', 'latex');
%     set(ht2, 'FontSize', label_font_size);

    
end
%% save book figures
% generate FIG file
% disp('Hit any key to print the figure and continue.');
% pause
% saveas(gcf, [figure_name,'_matlab'], 'fig');

% print 
% laprint(gcf, [figure_name,'_tex'],'scalefonts','off');

end