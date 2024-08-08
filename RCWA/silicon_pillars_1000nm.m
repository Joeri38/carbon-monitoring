clc;clear all;

addpath('./reticolo_allege');
addpath('reticolo_allege');

input = struct;
input.angle_delta = 0;
input.angle_theta = 0;
input.substrate_material = 'SiO2';
input.superstrate_material = 'air';
input.num_fourier = 12; % OG: 12
input.material_inside_circle = 'Si';
input.material_outside_circle = 'air';
input.cent_x = 0;
input.cent_y = 0;
input.patterned_layer_t_range = 1e-6; % [m] membrane thickness
input.period_range = 0.45e-6; % [m] OG: 0.22e-6
input.staircase_N = 20;

res = 100;
input.diameter_range = linspace(0.1e-6, 0.45e-6, res); 
input.lambda_range = 4.8e-6; % [m] 3.5 um
input.lambda0 = 4.8e-6; % [m] 3.5 um
input.debug = false;
input.multiplex = true;

total_tic = tic;
[all_meta_atoms,completed_jobs] = get_square_meta_atom_GD(input);

% to plot supercell: 
%get_supercell_meta_atom_GD.m %line 466: parfor
%get_supercell_meta_atom_GD.m %line 523: settings.debug = true;

%%
diameter = transpose(input.diameter_range);
phase = mod(angle(completed_jobs.amplitude_TE),2*pi); % [rad]
transmission = abs(completed_jobs.amplitude_TE).^2; % [1]
z_real = real(completed_jobs.amplitude_TE);
z_imag = imag(completed_jobs.amplitude_TE);
n_eff = completed_jobs.max_real_n_eff;

save_data = true;
if save_data
    save("library/pillars - 1000nm/squares_z_real_" + num2str(input.lambda0*1e6) + "um.mat", 'z_real');
    save("library/pillars - 1000nm/squares_z_imag_" + num2str(input.lambda0*1e6) + "um.mat", 'z_imag');
end

total_time = toc(total_tic);
fprintf('Total time elapsed: %g s, time per job: %g s\n', total_time);

figure;

subplot(1,2,1);
plot(input.diameter_range*1e6,phase(1:res)/(2*pi));
%plot(input.diameter_range*1e6,z_real(1:res));
xlabel('Diameter 1 [um]');
ylabel('Phase/2pi');
title("Lambda " + num2str(input.lambda0*1e6) + "um, period " + num2str(input.period_range*1e9) + "nm, height " + num2str(input.patterned_layer_t_range*1e9) + "nm");
%ylim([0, 2.05]);

subplot(1,2,2);
plot(input.diameter_range*1e6,transmission(1:res));
%plot(input.diameter_range*1e6,z_imag(1:res));
xlabel('Diameter 1 [um]');
ylabel('Transmission [%]');
title("Lambda " + num2str(input.lambda0*1e6) + "um, period " + num2str(input.period_range*1e9) + "nm, height " + num2str(input.patterned_layer_t_range*1e9) + "nm");
ylim([0, 1.05]);

saveas(gcf, "figs/silicon pillars 1000nm/squares_lamb=" + num2str(input.lambda0*1e6) + "um_p=" + num2str(input.period_range*1e9) + "nm_h=" + num2str(input.patterned_layer_t_range*1e9) + "nm.fig");

%% Estimate the amplitude based on the dominant bloch mode
neff = completed_jobs.max_real_n_eff;
k0L = (2*pi/input.lambda0)*input.patterned_layer_t_range;
n_substrate = get_refractive_index(input.substrate_material,input.lambda0*1e6);
n_superstrate = get_refractive_index(input.superstrate_material,input.lambda0*1e6);
est_transmission = trans_fn(neff,k0L,n_substrate,n_superstrate);
est_phases = angle(est_transmission); % [rad]
est_transmission = abs(est_transmission).^2; % [1]

%{
figure;%('units','normalized','position',[0,0,1,1]);

subplot(1,2,1);
plot(input.diameter1_range*1e6,mod(est_phases,2*pi)/pi,'-','linewidth',2);
xlabel('Diameter [um]');
ylabel('Phase [rad]');
title('dominant Bloch mode est.');
set(gca,'Fontsize',16);
set(gca,'Linewidth',2);
box on;


subplot(1,2,2);
plot(input.diameter1_range*1e6,100*est_transmission,'-','linewidth',2);
xlabel('Diameter [um]');
ylabel('Transmission [%]');
title('dominant Bloch mode est');
set(gca,'Fontsize',16);
set(gca,'Linewidth',2);
ylim([0, 105]);
box on;

%}

%% Auxillary functions
function transmission = trans_fn(n1_eff,k0L,n_substrate,n_superstrate)
% Gives the complex transmission coefficient for a medium 
% of effective index n1_eff, vacuum optical path lens
% k0L, on a substrate of index n_substrate, and below
% a superstrate of index n_superstrate.

% params:
% -n1_eff [1]: effective index of patterned layer. can be vector.
% -k0L [rad]: vacuum optical path length of patterned layer. must be either
% a single value or the same size as n1_eff.
% -n_substrate [1]: substrate index. must be a single value 
% or the same size as n1_eff.
% -n_superstrate [1]: superstrate index. must be a single value 
% or the same size as n1_eff.

% returns:
% -transmission [1]: complex transmission amplitude in the superstrate so
% that |transmission|^2 is the efficiency.

size_neff = size(n1_eff);
assert(length(k0L)==1 || all(size_neff == size(k0L)),'k0L must be same size as n1_eff');
assert(length(n_substrate)==1 || all(size_neff == size(n_substrate)),...
    'n_substrate must be same size as n1_eff');
assert(length(n_superstrate)==1 ||all(size_neff == size(n_superstrate)),...
    'n_superstrate must be same size as n1_eff');
c_t12 = 2*n_substrate./(n_substrate + n1_eff);
c_t23 = 2*n1_eff./(n1_eff + n_superstrate);
c_r21 = (n_substrate - n1_eff)./(n_substrate + n1_eff);
c_r23 = (n_superstrate - n1_eff)./(n_superstrate + n1_eff);
delta = n1_eff.*k0L;
transmission = sqrt(n_superstrate./n_substrate).*c_t12.*c_t23.*exp(1i*delta)./(1-c_r21.*c_r23.*exp(2i*delta));
end