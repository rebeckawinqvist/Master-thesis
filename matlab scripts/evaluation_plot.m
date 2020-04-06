set(0,'defaulttextInterpreter','latex') 
lw = 2;
sz = 20;
close all;

%% 100 trajs
ntrajs = 72;
N = 10;
example = "E";

fn_proj = sprintf("ex%s_eval_projNN_ntrajs_%s_N_%s_alltrajs.csv", example, int2str(ntrajs), int2str(N));
fn_lqrproj = sprintf("ex%s_eval_lqr_projNN_ntrajs_%s_N_%s_alltrajs.csv", example, int2str(ntrajs), int2str(N));
fn_noproj = sprintf("ex%s_eval_noprojNN_ntrajs_%s_N_%s_alltrajs.csv", example, int2str(ntrajs), int2str(N));
fn_mpc = sprintf("ex%s_eval_mpc_ntrajs_%s_N_%s_alltrajs.csv", example, int2str(ntrajs), int2str(N));

proj_costs = readmatrix(fn_proj);
lqrproj_costs = readmatrix(fn_lqrproj);
noproj_costs = readmatrix(fn_noproj);
mpc_costs = readmatrix(fn_mpc);

if example == "E"
    proj_costs = proj_costs(1:70,1:end);
    lqrproj_costs = lqrproj_costs(1:70,1:end);
    noproj_costs = noproj_costs(1:70,1:end);
    mpc_costs = mpc_costs(1:70,1:end);
end

ntrajs = 70;
x = [1:1:ntrajs];


f1 = figure(1);
hold on;
grid on; grid minor;
plot(x, proj_costs, '-', 'LineWidth', lw);
plot(x, lqrproj_costs, '-.', 'LineWidth', lw);
plot(x, noproj_costs, '--', 'LineWidth', lw);
plot(x, mpc_costs, ':', 'LineWidth', lw);
legend('PNN', 'LQR-PNN', 'BBNN', 'Implicit MPC');
xlabel('Trajectory index')
ylabel('Normalized control cost')

fn1 = sprintf("ex%s_traj_eval_%s_plot.png", example, int2str(ntrajs))
saveas(f1, fn1)

% f2 = figure(2);
% hold on;
% grid on; grid minor;
% scatter(x, proj_costs, sz,'filled', 'o');
% scatter(x, lqrproj_costs, sz, 'filled', '^');
% scatter(x, noproj_costs, sz, 'filled', 's');
% scatter(x, mpc_costs, sz, 'filled', 'd');
% legend('PNN', 'LQR-PNN', 'BBNN', 'Implicit MPC');
% xlabel('Trajectory index')
% ylabel('Cost')
% 
% fn2 = sprintf("ex%s_traj_eval_%s_scatter.png", example, int2str(ntrajs))
% saveas(f2, fn2)

%% 1000 trajs
ntrajs = 574;
N = 10;
example = "E";

fn_proj = sprintf("ex%s_eval_projNN_ntrajs_%s_N_%s_alltrajs.csv", example, int2str(ntrajs), int2str(N));
fn_lqrproj = sprintf("ex%s_eval_lqr_projNN_ntrajs_%s_N_%s_alltrajs.csv", example, int2str(ntrajs), int2str(N));
fn_noproj = sprintf("ex%s_eval_noprojNN_ntrajs_%s_N_%s_alltrajs.csv", example, int2str(ntrajs), int2str(N));
fn_mpc = sprintf("ex%s_eval_mpc_ntrajs_%s_N_%s_alltrajs.csv", example, int2str(ntrajs), int2str(N));

proj_costs = readmatrix(fn_proj);
lqrproj_costs = readmatrix(fn_lqrproj);
noproj_costs = readmatrix(fn_noproj);
mpc_costs = readmatrix(fn_mpc);

if example == "E"
    proj_costs = proj_costs(1:500,1:end);
    lqrproj_costs = lqrproj_costs(1:500,1:end);
    noproj_costs = noproj_costs(1:500,1:end);
    mpc_costs = mpc_costs(1:500,1:end);
end

ntrajs = 500;
x = [1:1:ntrajs];

f3 = figure(3);
hold on;
grid on; grid minor;
plot(x, proj_costs, '-', 'LineWidth', lw);
plot(x, lqrproj_costs, '-.', 'LineWidth', lw);
plot(x, noproj_costs, '--', 'LineWidth', lw);
plot(x, mpc_costs, ':');
legend('PNN', 'LQR-PNN', 'BBNN', 'Implicit MPC');
xlabel('Trajectory index')
ylabel('Normalized control cost')

fn3 = sprintf("ex%s_traj_eval_%s_plot.png", example, int2str(ntrajs))
saveas(f3, fn3)


% f4 = figure(4);
% hold on;
% grid on; grid minor;
% scatter(x, proj_costs, sz, 'filled', 'o');
% scatter(x, lqrproj_costs, sz, 'filled', '^');
% scatter(x, noproj_costs, sz, 'filled', 's');
% scatter(x, mpc_costs, sz, 'filled', 'd');
% legend('Proj', 'LQR', 'NoProj', 'MPC');
% xlabel('Trajectory index')
% ylabel('Cost')
% 
% fn4 = sprintf("ex%s_traj_eval_%s_scatter.png", example, int2str(ntrajs))
% saveas(f4, fn)

%% Example A
example = "A";
ntrajs = 100;
N = 3;

fn_proj = sprintf("ex%s_eval_projNN_ntrajs_%s_N_%s_alltrajs.csv", example, int2str(ntrajs), int2str(N));
fn_lqrproj = sprintf("ex%s_eval_lqr_projNN_ntrajs_%s_N_%s_alltrajs.csv", example, int2str(ntrajs), int2str(N));
fn_noproj = sprintf("ex%s_eval_noprojNN_ntrajs_%s_N_%s_alltrajs.csv", example, int2str(ntrajs), int2str(N));
fn_mpc = sprintf("ex%s_eval_mpc_ntrajs_%s_N_%s_alltrajs.csv", example, int2str(ntrajs), int2str(N));

proj_costs = readmatrix(fn_proj);
lqrproj_costs = readmatrix(fn_lqrproj);
noproj_costs = readmatrix(fn_noproj);
mpc_costs = readmatrix(fn_mpc);

x = [1:1:ntrajs];

f5 = figure(5);
hold on;
grid on; grid minor;
plot(x, proj_costs, '-', 'LineWidth', lw);
plot(x, lqrproj_costs, '-.', 'LineWidth', lw);
plot(x, noproj_costs, '--', 'LineWidth', lw);
plot(x, mpc_costs, ':', 'LineWidth', lw);
legend('PNN', 'LQR-PNN', 'BBNN', 'Implicit MPC');
xlabel('Trajectory index')
ylabel('Normalized control cost')

fn5 = sprintf("ex%s_traj_eval_%s_plot.png", example, int2str(ntrajs))
saveas(f5, fn5)

% f6 = figure(6);
% hold on;
% grid on; grid minor;
% scatter(x, proj_costs, sz,'filled', 'o');
% scatter(x, lqrproj_costs, sz, 'filled', '^');
% scatter(x, noproj_costs, sz, 'filled', 's');
% scatter(x, mpc_costs, sz, 'filled', 'd');
% legend('PNN', 'LQR-PNN', 'BBNN', 'Implicit MPC');
% xlabel('Trajectory index')
% ylabel('Cost')
% 
% fn6 = sprintf("ex%s_traj_eval_%s_scatter.png", example, int2str(ntrajs))
% saveas(f6, fn6)

