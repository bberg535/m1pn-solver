function starmap_lattice(N, n, filt, outdir, starmap_dir)
%STARMAP_LATTICE  StaRMAP (Seibold/Frank, v2.0, github.com/starmap-project/starmap) on the lattice-paul problem
%   (Brunner/Holloway checkerboard as in StaRMAP's starmap_2d_ex_lattice.m: P_N, staggered grid, extrapolation
%   boundary, psi_0 = 0, T = 3.2). Writes summary.json (wall time of the solver, mass, absorption rate at T, total
%   absorption by the trapezoidal rule over 321 outputs) and final.csv (x, y, rho) into OUTDIR.
%   FILT = 0: plain P_N; 1: StaRMAP's exponential filter of starmap_2d_ex_lattice_filter.m (f_eff = 5, p = 2, substep).
addpath(starmap_dir);
global ACC
ACC = struct('t', [], 'rate', [], 'mass', [], 'outdir', outdir, 'nt', 321);
if ~exist(outdir, 'dir'), mkdir(outdir); end
prob = struct('name', 'Lattice', 'closure', 'P', 'n_mom', N, 'sigma_a', @sigma_a, 'sigma_s0', @sigma_s0, ...
    'source', @source, 'ax', [0 7 0 7], 'n', [n n], 'bc', [1 1], 't_plot', linspace(0, 3.2, ACC.nt), ...
    'output', @output);
if filt
    prob.filter = 'exp'; prob.filterfunction = @exp_filter; prob.f_position = 'substep';
end
par = starmap_init(prob);
t0 = tic; starmap_solver(par); wall = toc(t0);
absorbed = trapz(ACC.t, ACC.rate);
fid = fopen(fullfile(outdir, 'summary.json'), 'w');
fprintf(fid, ['{"code":"StaRMAP 2.0","pn_order":%d,"cells":%d,"filter":%d,"wall_seconds":%.6f,', ...
    '"threads":%d,"mass":%.15g,"rate":%.15g,"absorbed":%.15g}\n'], N, n, filt, wall, maxNumCompThreads, ...
    ACC.mass(end), ACC.rate(end), absorbed);
fclose(fid);
end

function f = absorber(x, y)
cx = ceil(x); cy = ceil(y);
f = (ceil((cx + cy) / 2) * 2 == (cx + cy)) .* (1 < cx & cx < 7 & 1 < cy & cy - 2 * abs(cx - 4) < 4);
end
function f = sigma_a(x, y), f = 10 * absorber(x, y); end
function f = sigma_s0(x, y), f = 1 - absorber(x, y); end
function f = source(x, y), f = double(3 < x & x < 4 & 3 < y & y < 4); end
function f = exp_filter(par, dt, k)
p = 2; f_eff = 5;
f_b = -f_eff / log(exp(log(eps) * (par.n_mom / (par.n_mom + 1))^p));
f = exp(log(eps) * (k / (par.n_mom + 1)).^p).^(dt * f_b);
end
function output(par, x, y, U, step)
global ACC
[X, Y] = ndgrid(x, y);
dA = (x(2) - x(1)) * (y(2) - y(1));
ACC.t(end + 1) = par.t_plot(step);
ACC.rate(end + 1) = sum(sigma_a(X, Y) .* U, 'all') * dA;
ACC.mass(end + 1) = sum(U, 'all') * dA;
if step == ACC.nt
    writematrix([X(:) Y(:) U(:)], fullfile(ACC.outdir, 'final.csv'));
end
end
