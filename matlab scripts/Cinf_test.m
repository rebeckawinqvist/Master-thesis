
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
save_matrices = true;
save_figures = true;

%% Example A
% matrices of LTI dynamics 
% x(k+1) = A*x(k) + B*u(k)
A=[1 1 ; 0 1];
B=[0 ; 1];

% create model in MPT3 interface
model = LTISystem('A',A,'B',B);

% constraints on inputs and states
model.u.min = -1;
model.u.max = 1;
model.x.min = [-5 ; -5];
model.x.max = [5 ; 5];

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

figure(1);
% plot the constraint set
plot(X,'color',[0.7 0.7 0.7],'linewidth',line_width);
hold on
grid on
% plot the intermediate sets
hp = plot(Piter,'color',[0.7 0.7 0.7]);
% plot the invariant set
plot(Cinf,'color',[0.5 0.5 0.5],'linewidth',line_width)
axis([model.x.min(1),model.x.max(1),model.x.min(2),model.x.max(2)])
title('ExA: C_\infty ')

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


% write system dynamics to file
if save_matrices
    dlmwrite([pwd '\ExA\exA_A.csv'], A)
    dlmwrite([pwd '\ExA\exA_B.csv'], B)
    dlmwrite([pwd '\ExA\exA_xlb.csv'], model.x.min)
    dlmwrite([pwd '\ExA\exA_xub.csv'], model.x.max)
    dlmwrite([pwd '\ExA\exA_ulb.csv'], model.u.min)
    dlmwrite([pwd '\ExA\exA_uub.csv'], model.u.max)
    dlmwrite([pwd '\ExA\exA_cinfH.csv'], Cinf.H)
    dlmwrite([pwd '\ExA\exA_cinfV.csv'], Cinf.V)
end

if save_figures
    saveas(figure(1),[pwd '\ExA\exA_Cinf.fig']);
end


% ------------------ Example B ------------------

% matrices of LTI dynamics 
% x(k+1) = A*x(k) + B*u(k)
A=[1 1 ; 0 1];
B=[0 ; 1];

% create model in MPT3 interface
model = LTISystem('A',A,'B',B);

% constraints on inputs and states
model.u.min = -5;
model.u.max = 5;
model.x.min = [-10 ; -10];
model.x.max = [10 ; 10];

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

figure(2)
% plot the constraint set
plot(X,'color',[0.7 0.7 0.7],'linewidth',line_width);
hold on
grid on
% plot the intermediate sets
hp = plot(Piter,'color',[0.7 0.7 0.7]);
% plot the invariant set
plot(Cinf,'color',[0.5 0.5 0.5],'linewidth',line_width)
axis([model.x.min(1),model.x.max(1),model.x.min(2),model.x.max(2)])
title('ExB: C_\infty ')

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


% write system dynamics to file
if save_matrices
    dlmwrite([pwd '\ExB\exB_A.csv'], A)
    dlmwrite([pwd '\ExB\exB_B.csv'], B)
    dlmwrite([pwd '\ExB\exB_xlb.csv'], model.x.min)
    dlmwrite([pwd '\ExB\exB_xub.csv'], model.x.max)
    dlmwrite([pwd '\ExB\exB_ulb.csv'], model.u.min)
    dlmwrite([pwd '\ExB\exB_uub.csv'], model.u.max)
    dlmwrite([pwd '\ExB\exB_cinfH.csv'], Cinf.H)
    dlmwrite([pwd '\ExB\exB_cinfV.csv'], Cinf.V)
end

if save_figures
    saveas(figure(2),[pwd '\ExB\exB_Cinf.fig']);
end


% ------------------ Example C ------------------
% matrices of LTI dynamics 
% x(k+1) = A*x(k) + B*u(k)
A=[1 1 ; 0 1];
B=[0 ; 1];

% create model in MPT3 interface
model = LTISystem('A',A,'B',B);

% constraints on inputs and states
model.u.min = -1;
model.u.max = 1;
%model.x.min = [-inf ; -inf];
%model.x.max = [inf ; inf];

% constraint sets represented as polyhedra
% X = Polyhedron('lb',model.x.min,'ub',model.x.max);
% U = Polyhedron('lb',model.u.min,'ub',model.u.max);

% write system dynamics to file
if save_matrices
    dlmwrite([pwd '\ExC\exC_A.csv'], A)
    dlmwrite([pwd '\ExC\exC_B.csv'], B)
    dlmwrite([pwd '\ExC\exC_xlb.csv'], model.x.min)
    dlmwrite([pwd '\ExC\exC_xub.csv'], model.x.max)
    dlmwrite([pwd '\ExC\exC_ulb.csv'], model.u.min)
    dlmwrite([pwd '\ExC\exC_uub.csv'], model.u.max)
end


% ------------------ Example D ------------------

% matrices of LTI dynamics 
% x(k+1) = A*x(k) + B*u(k)
A=[1 1 ; 0 1];
B=[0 ; 1];

% create model in MPT3 interface
model = LTISystem('A',A,'B',B);

% constraints on inputs and states
model.u.min = -0.5;
model.u.max = 0.5;
model.x.min = [-5 ; -5];
model.x.max = [5 ; 5];

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

figure(3)
% plot the constraint set
plot(X,'color',[0.7 0.7 0.7],'linewidth',line_width);
hold on
grid on
% plot the intermediate sets
hp = plot(Piter,'color',[0.7 0.7 0.7]);
% plot the invariant set
plot(Cinf,'color',[0.5 0.5 0.5],'linewidth',line_width)
axis([model.x.min(1),model.x.max(1),model.x.min(2),model.x.max(2)])
title('ExD: C_\infty ')

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


% write system dynamics to file
if save_matrices
    dlmwrite([pwd '\ExD\exD_A.csv'], A)
    dlmwrite([pwd '\ExD\exD_B.csv'], B)
    dlmwrite([pwd '\ExD\exD_xlb.csv'], model.x.min)
    dlmwrite([pwd '\ExD\exD_xub.csv'], model.x.max)
    dlmwrite([pwd '\ExD\exD_ulb.csv'], model.u.min)
    dlmwrite([pwd '\ExD\exD_uub.csv'], model.u.max)
    dlmwrite([pwd '\ExD\exD_cinfH.csv'], Cinf.H)
    dlmwrite([pwd '\ExD\exD_cinfV.csv'], Cinf.V)
end

if save_figures
    saveas(figure(3),[pwd '\ExD\exD_Cinf.fig']);
end



% ------------------ Example E ----------------------------
% matrices of LTI dynamics 
% x(k+1) = A*x(k) + B*u(k)
A=[0.7 -0.1 0.0 0.0 ; 0.2 -0.5 0.1 0.0 ; 0.0 0.1 0.1 0.0 ; 0.5 0.0 0.5 0.5]
B=[0.0 0.1 ; 0.1 1.0 ; 0.1 0.0 ; 0.0 0.0]


% create model in MPT3 interface
model = LTISystem('A',A,'B',B);

% constraints on inputs and states
model.u.min = [-5 ; -5];
model.u.max = [5 ; 5];
model.x.min = [-6 ; -6 ; -1 ; -0.5];
model.x.max = [6 ; 6 ; 1 ; 0.5];

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

% figure(5);
% % plot the constraint set
% plot(X,'color',[0.7 0.7 0.7],'linewidth',line_width);
% hold on
% grid on
% % plot the intermediate sets
% hp = plot(Piter,'color',[0.7 0.7 0.7]);
% % plot the invariant set
% plot(Cinf,'color',[0.5 0.5 0.5],'linewidth',line_width)
% axis([model.x.min(1),model.x.max(1),model.x.min(2),model.x.max(2)])
% title('ExE: C_\infty ')
% 
% set(gca,'LineWidth',axeswidth)
% set(gca,'FontSize', tick_font_size);
% xt = transpose(-10:5:10);
% yt = transpose(-10:5:10);
% set(gca,'XTick',xt);
% set(gca,'YTick',yt);
% set(gca,'YTickLabel',num2str(xt));
% set(gca,'XTickLabel',num2str(yt));
% 
% hx1 = xlabel('$x_1$', 'interpreter', 'latex');
% set(hx1, 'FontSize', label_font_size);
% hy1 = ylabel('$x_2$', 'interpreter', 'latex');
% set(hy1, 'FontSize', label_font_size);
% 
% ht1=text(7,-8,'$\mathcal{X}$', 'interpreter', 'latex');
% set(ht1, 'FontSize', label_font_size);
% ht2=text(-1,-0.5,'$\mathcal{C}_{\infty}$', 'interpreter', 'latex');
% set(ht2, 'FontSize', label_font_size);
% 
% border = 1;
% axis([model.x.min(1)-border,model.x.max(1)+border,model.x.min(2)-border,model.x.max(2)+border])


% write system dynamics to file
if save_matrices
    dlmwrite([pwd '\ExE\exE_A.csv'], A)
    dlmwrite([pwd '\ExE\exE_B.csv'], B)
    dlmwrite([pwd '\ExE\exE_xlb.csv'], model.x.min)
    dlmwrite([pwd '\ExE\exE_xub.csv'], model.x.max)
    dlmwrite([pwd '\ExE\exE_ulb.csv'], model.u.min)
    dlmwrite([pwd '\ExE\exE_uub.csv'], model.u.max)
    dlmwrite([pwd '\ExE\exE_cinfH.csv'], Cinf.H)
    dlmwrite([pwd '\ExE\exE_cinfV.csv'], Cinf.V)
end

if save_figures
    %saveas(figure(5),[pwd '\ExE\exE_Cinf.fig']);
end
