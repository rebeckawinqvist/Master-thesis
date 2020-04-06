set(0,'defaulttextInterpreter','latex')
close all;

example = "E";
x_act = readmatrix(sprintf("ex%s_comp_act_train_samples.csv", example));
x = readmatrix(sprintf("ex%s_comp_train_samples.csv", example));

proj_mse = readmatrix(sprintf("ex%s_proj_mse_comp.csv", example));
lqr_mse = readmatrix(sprintf("ex%s_lqr_proj_mse_comp.csv", example));
noproj_mse = readmatrix(sprintf("ex%s_noproj_mse_comp.csv", example));

proj_nmse = readmatrix(sprintf("ex%s_proj_nmse_comp.csv", example));
lqr_nmse = readmatrix(sprintf("ex%s_lqr_proj_nmse_comp.csv", example));
noproj_nmse = readmatrix(sprintf("ex%s_noproj_nmse_comp.csv", example));

    
sz = 50;

lw = 2;

f = figure(1);
hold on;
grid on; grid minor;
plot(x_act, 10*log10(proj_nmse), '-o', 'LineWidth', lw);
plot(x_act, 10*log10(lqr_nmse), '-.^', 'LineWidth', lw);
plot(x_act, 10*log10(noproj_nmse), '--s', 'LineWidth', lw);
legend('PNN', 'LQR-PNN', 'BBNN');
xlabel('Number of training samples')
ylabel('NMSE [dB]')

f2 = figure(2);
hold on;
grid on; grid minor;
plot(x_act, 10*log10(proj_mse), '-o', 'LineWidth', lw);
plot(x_act, 10*log10(lqr_mse), '-.^', 'LineWidth', lw);
plot(x_act, 10*log10(noproj_mse), '--s', 'LineWidth', lw);
legend('PNN', 'LQR-PNN', 'BBNN');
xlabel('Number of training samples')
ylabel('10log10(Mean MSE)')

f3 = figure(3);
hold on;
grid on; grid minor;
plot(x_act, (proj_nmse), '-o');
plot(x_act, (lqr_nmse), '-.^');
plot(x_act, (noproj_nmse), '--s');
legend('PNN', 'LQR-PNN', 'BBNN');
xlabel('Number of training samples')
ylabel('NMSE')

fn1 = sprintf("ex%s_lognmse.png", example);
saveas(f, fn1)

fn2 = sprintf("ex%s_logmse.png", example);
saveas(f2, fn2)

