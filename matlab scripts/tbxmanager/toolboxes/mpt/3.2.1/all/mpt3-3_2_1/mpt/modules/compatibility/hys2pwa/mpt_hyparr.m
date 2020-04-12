%===============================================================================
%
% Title:        hyparr
%                                                             
% Project:      Transformation of HYSDEL model into PWA model
%
% Purpose:      Dervive hyperplane arrangement
%
% Input:        Hyp: hyperplanes Hyp.A(i,:)*x = Hyp.B(i), for i = 1...rows(Hyp.A)
%                    that define a hyperplane arrangement
%               P: constraints P.A*x <= P.B define polyhedron
%                  P=[] is possible 
%               dom: constraints dom.A*x <= dom.B defines the domain
%                    these constraints are only used if dom.constr exists
%               lpsolver
%               verbose
%               minR: minimal required radius of Chebycheff ball of polyhedra
%
% Output:       delta: hyperplane arrangement, or the markings of the regions 
%                      generated by the hyperplanes Hyp, where delta(:,j) 
%                      represents the j-th region of the arrangement as a 
%                      {-1, 1} vector. 
%                      Only the markings of the regions having feasible points 
%                      in P and dom are returned.
%
%               Example: 
%                   delta = [1  1 -1;
%                            1 -1 -1] 
%                   means that there were 2 hyperplanes and that they induce 3 
%                   polyhedra: 
%                   1. A x <= B;
%                   2. A(1,:) <= B(1),  A(2,:) >= B(2);
%                   3. A(1,:) >= B(1),  A(2,:) >= B(2);
%
%               Note: This definition is contrary to the one in optMerge 
%                     (and Ziegler)
%
% Comments:     By default, the reverse search tool by Komei Fukuda is currently 
%               not used, as it does not work realiably for large problems.
%               Instead, the feasibility of the regions is checked by solving LPs.
%               The efficiency is improved by the following means:
%               * if dom.constr=1, only regions with feasible points in dom are 
%                 considered
%               * large hyperplane arrangements are stored in the function 
%               * in a last step, only the regions with feasible points in P are 
%                 returned (thus allowing the storage of already computed
%                 hyperplane arrangements)
%
% Authors:      Tobias Geyer <geyer@control.ee.ethz.ch>, Fabio Torrisi
                                                                      
% History:      date        subject                                       
%               2002.09.23  initial version
%               2003.05.09  code rewritten, storage of computed hyperplane arrangements
%
% Requires:     zonotope
%
% Contact:      Tobias Geyer
%               Automatic Control Laboratory
%               ETH Zentrum, CH-8092 Zurich, Switzerland
%
%               geyer@aut.ee.ethz.ch
%
%               Comments and bug reports are highly appreciated
%
%===============================================================================

function delta = mpt_hyparr(Hyp, P, dom, lpsolver, verbose, minR)

global mptOptions

if ~isstruct(mptOptions)
    mpt_error;
end

% hyperplane arrangement as persistent 
% = local to the function hyparr yet their values are 
%   retained in memory between calls to the function.  
persistent HA

if nargin==0,
    % clear persistent variable and exit if no input arguments are given. this
    % is needed by mpt_sys(), otherwise we can get different results, and even
    % errors, when running the function subsequently on different hysdel models
    clear HA
    return
end


% complement the inputs
if isempty(P), P.A = []; P.B = []; end;
if ~isfield(dom, 'constr'), dom.constr=0; end;
if nargin < 4, lpsolver = 0; end;
if nargin < 5, verbose = 0; end;
if nargin < 6, minR = 0; end;

% LP solver:
% lpsolver=0: uses E04MBF.m
% lpsolver=1: uses linprog.m (MATLAB)
% lpsolver=2: uses CPLEX
lpsolver = mptOptions.lpsolver;


% normalize hyperplanes A*x=B
norm = sqrt(sum(Hyp.A.*Hyp.A,2));       % using: Euclidean norm
% norm = max(abs(Hyp.A),[],2);          % optional: Infinity norm
if min(norm)==0, error('There is at least one badly defined hyperplane!'); end
Hyp.A = Hyp.A .* kron(1./norm, ones(1,size(Hyp.A,2)));
Hyp.B = Hyp.B./norm;

% restrict hyperplane arrangement a priori to dom.H*[xr; ur] <= dom.K?
if dom.constr 
    % add the bounds on xr-ur space (given at the declaration of
    % the xr and ur variables in hysdel) as constraints.
    Hdom = dom.H;
    Kdom = dom.K;
else
    % don't add them
    Hdom = [];
    Kdom = [];
end;

delta = [];
nc = length(Hyp.B); % number of constraints / hyperplanes

% check out different approaches:
% delta = hyparr_test(Hyp, P, dom, verbose, -1);

if nc >= 4

    % we have at least four hyperplanes
    % reusing already computed hyperplane arrangements is thus very reasonable
    
    % is Hyp.A*x=Hyp.B an entry of HA?
    % (= has the hyperplane arrangement for Hyp.A*x=Hyp.B been already computed?)
    i = 1;
    while i <= length(HA)
        if length(HA{i}.B) == nc
            % same number of hyperplanes
            if all( HA{i}.B == Hyp.B )
                % B vectors are the same
                if all (HA{i}.A == Hyp.A )
                    % also A matrices are the same
                    % --> we have found an entry for A*x=B
                    delta = HA{i}.delta;
                    %if verbose, fprintf('  found hyperplane arrangement with %1i hyperplanes\n', length(Hyp.B)); end;
                    if verbose==2, fprintf('f(%1i) ', length(Hyp.B)); end;
                    break
                end;
            end;
        end;
        i = i+1;
    end;
    
    if isempty(delta)
        % derive hyperplane arrangement restricted to the domain (if set)
        % using the iterative approach (efficient for large hyperplane arrangements)
        t0 = clock;
        delta = iter_lp(Hyp.A, Hyp.B, Hdom, Kdom, lpsolver, minR);
        if verbose==2, fprintf('c(%0.2fs) ', etime(clock, t0)); end;
        
        % store in HA
        HA{end+1}.A = Hyp.A;
        HA{end}.B = Hyp.B;
        HA{end}.delta = delta;
    end;
    
    % extract the polyhedra that are within the polyhedron P,
    % i.e. restrict polyhedra a posteriori to P.A*[xr; ur] <= P.B
    t0 = clock;
%     i = 1;
%     while i <= size(delta,2)
%         % consider polyhedron given by the marking delta(i,:)
%         m = delta(:,i);
%         
%         % get the center x and the radius r of the polyhedron 
%         [x,r] = polyinnerball([diag(m)*Hyp.A; P.A],[diag(m)*Hyp.B; P.B], lpsolver);
%         if r >= minR, i = i+1;
%         else          delta(:,i) = []; 
%         end
%     end
    keepInd = [];
    for i = 1:size(delta,2)
        % consider polyhedron given by the marking delta(i,:)
        m = delta(:,i);
        
        % get the center x and the radius r of the polyhedron 
        [x,r] = polyinnerball([diag(m)*Hyp.A; P.A],[diag(m)*Hyp.B; P.B], lpsolver);
        if r >= minR, keepInd(end+1) = i; end;
    end
    delta = delta(:,keepInd);
    if verbose==2, fprintf('r(%0.2fs) ', etime(clock, t0)); end;

else
    
    % we have less than four hyperplanes
    % it is faster to recompute the hyperplane arrangement each time newly
    % taking into account the bounds given by P and dom 
    delta = std_lp(Hyp.A, Hyp.B, [P.A; Hdom], [P.B; Kdom], lpsolver, minR);
    
end;



% --------------------------------------------------------------------

function delta = std_lp(A, B, H, K, lpsolver, minR)
% given the hyperplanes A*x = B
% enumerate all regions in the polyhedron H*x<=K
delta = [];
nc = length(B); % number of constraints / hyperplanes
for i = 0:2^nc-1,
    m = + ones(nc,1) - 2*(dec2bin(i,nc)-'0')';    % derive the i-th marking
    
    % get the center x and the radius r of the polyhedron
    [x,r] = polyinnerball([diag(m)*A; H],[diag(m)*B; K], lpsolver);
    if r >= minR, delta(:,end+1) = m; end
end




% --------------------------------------------------------------------

function D = iter_lp(A, B, H, K, lpsolver, minR)
% given the hyperplanes A*x = B
% enumerate all regions *iteratively* in the polyhedron H*x<=K

% idea:
% build hyperplane arrangement step by step adding one hyperplane per step.
% Thus, each polyhedron is possibly cut into two. Only the feasible ones are 
% kept at every step. Feasible means, that the radius of their Chebycheff
% ball is larger than minR

% initial step: first hyperplane
D = [];

% '-'
[x,r] = polyinnerball([(-1)*A(1,:); H], [-B(1); K], lpsolver);
if r >= minR, D(:,end+1) = -1; end

% '+'
[x,r] = polyinnerball([A(1,:); H], [B(1); K], lpsolver);
if r >= minR, D(:,end+1) = +1; end

% % second step: remaining hyperplanes
% for i = 2:length(B),
%     % add i-th hyperplane to the markings in D
%     D_augm = [];
%     for j = 1:size(D,2)
%         % add i-th hyperplane to the j-th polyhedron in D
%         m = D(:,j);
%         % -
%         [x,r] = polyinnerball([diag([m;-1])*A(1:i,:); H], [diag([m;-1])*B(1:i); K], lpsolver);
%         if r >= minR, D_augm(:,end+1) = [m;-1]; end
%         % +
%         [x,r] = polyinnerball([diag([m;+1])*A(1:i,:); H], [diag([m;+1])*B(1:i); K], lpsolver);
%         if r >= minR, D_augm(:,end+1) = [m;+1]; end
%     end;
%     D = D_augm;
% end

% second step: remaining hyperplanes
for i = 2:length(B),
    
    % reserve memory for D_augm to speed up the algorithm
    D_augm = NaN*ones(i,2*size(D,2));
    len = 0;
    
    % add i-th hyperplane to the markings in D
    for j = 1:size(D,2)
        
        % add i-th hyperplane to the j-th polyhedron in D
        m = D(:,j);
        
        % -
        [x,r] = polyinnerball([diag([m;-1])*A(1:i,:); H], [diag([m;-1])*B(1:i); K], lpsolver);
        if r >= minR, 
            len = len+1;
            D_augm(:,len) = [m;-1]; 
        end
        
        % +
        [x,r] = polyinnerball([diag([m;+1])*A(1:i,:); H], [diag([m;+1])*B(1:i); K], lpsolver);
        if r >= minR, 
            len = len+1;
            D_augm(:,len) = [m;+1]; 
        end
    end;
    D = D_augm(:,1:len);
end

return








































% --------------------------------------------------------------------
% --------------------------------------------------------------------

function delta = lp_hyparr_old(A,B,H,K)
delta = [];
nx = size(A,2);
na = size(H,1);
nc = size(A,1);
wc = 2^nc;      % argh: go through all the combinations
for i = 0:wc-1,
    reg = + ones(nc,1) - 2*(dec2bin(i,nc)-'0')';    % derive the i-th marking
    
    %if is_feasible([diag(reg)*A;H],[diag(reg)*B;K]),% check whether the corresponding region is feasible
    %    delta = [delta, reg];
    
    % get the center x and the radius r of the polyhedron
    % solver: 0=NAG, 1=linprog, 2=cplex
    [x,r] = polyinnerball([diag(reg)*A;H],[diag(reg)*B;K], 0);
    if r > 0, delta = [delta, reg]; end
end




% --------------------------------------------------------------------

function delta = kf_hyparr(A,B,H,K)

% add the bounding constraints H, K
AA = [[A;H],-[B;K];[zeros(1,size(A,2)),1]];

% to avoid numerical problems
ind = find(abs(AA)<1e-10);
AA(ind) = 0;

% Play the dirty trick (rs_alltope accepts only integers)
m = min(abs(AA(find(AA)))); 
M = max(abs(AA(find(AA)))); 
% AA = round(1e5 * AA / m);
maxInt = 15500;
AA = round(maxInt/M * AA);

% regularize the matrix 
% 1. find the zero columns
zerocols = all(AA == 0,1);

% 2. find the repeated columns
nc = size(AA,1);
redrows = zeros(nc,1);
for i = 1:size(AA,1),
    for j = i+1:size(AA,1),
        if AA(i,:) == AA(j,:)
            redrows(j,1) = i; % store that row j = row i
        end
    end
end

% get markings
delta_tmp = zonotope(AA(find(redrows == 0),find(~zerocols)));
delta(find(redrows == 0),:) = delta_tmp;

% add back the rows that were removed
for i = 1:size(AA,1)
    if redrows(i,1) ~= 0,
        delta(i,:)=delta(redrows(i,1),:);
    end
end

% consider the lifted hyperplane arrangement should have '-1' in the last constraint
% therefore remove the entries with 1
delta(:,find(delta(end, :) == 1)) = [];

% remove the last line containing the added constraint
delta(end,:) = [];

% Consider only the regions that satisfy H and K
% therefore remove the entries with -1
for i = (size(A, 1) + 1):(size(A, 1) + size(H, 1))
    delta(:, find(delta(i, :) == -1)) = [];
end

% remove the markings corresponding to the bounding constraints
delta(end-length(K)+1:end,:)=[];


















function delta = hyparr_test(Hyp, P, dom, verbose, choice)

% there are two ways of computing that:
% 1. Checking all the possibilities
% 2. Using the tool from Komei Fukuda <http://www.cs.mcgill.ca/~fukuda/download/mink/>
% the variable choice decides which approach to use.

% Make sure, that there is not the same constraint twice

lpsolver = 0;

% choice"  
% -1: compare both methods
%     meaning of the output: computation time for method 0,
%                            computation time for method 1
%                            - if the results differ, else +
%  0: enumerate all combinations 
%     = no numerical problems, fast for small problems
%  1: use reverse search 
%     = possibly numerical problems, fast for large problems

if choice == -1
    
    fprintf('compare both hyperplane arrangements: ')
    
    % enumerate all combinations:
    t0 = clock;
    minR = 1e-10;
    %D_lp = lp_hyparr(Hyp,P,dom,verbose);
    D_lp = iter_lp(Hyp.A, Hyp.B, P.H, P.K, lpsolver, minR);
    %D_lp = std_lp(Hyp.A, Hyp.B, P.H, P.K, lpsolver, 0);
    fprintf('%0.2fs  ', etime(clock, t0));
    
    % plot solution
    color.Table{1} = [1:size(D_lp,2)];
    GL.Hi = Hyp.A; GL.Ki = Hyp.B;
    plotPolyMark((-1)*D_lp, color, GL, dom);
    
    % use reverse search:
    t0 = clock;
    D_kf = kf_hyparr(Hyp.A, Hyp.B, P.H, P.K);
    %D_kf = eff_hyparr(Hyp,P,dom,verbose);
    fprintf('%0.2fs  ', etime(clock, t0));

    % plot solution
    color.Table{1} = [1:size(D_kf,2)];
    GL.Hi = Hyp.A; GL.Ki = Hyp.B;
    plotPolyMark((-1)*D_kf, color, GL, dom);
    
    
    delta = D_lp;
      
    % are both markings the same?
    
    % i) do they have the same dimensions?
    if ~all(size(D_lp) == size(D_kf))
        fprintf('-\n')
        return
    end;
    
    % ii) reorder both markings such that in the first columns are the rows
    %     with most -1 entries using something like bubble sort
    if reorder(D_lp) == reorder(D_kf)
        fprintf('+\n')
    else
        fprintf('-\n')
    end;

elseif choice == 0
    
    %fprintf('\n')
    %nd = length(B);
    %fprintf('get hyperplane arrangements (by enumeration) of %1i d-variables ... ', nd)
    
    %t0 = clock;
    delta = lp_hyparr(A,B,H,K,dom,verbose);
    %fprintf(' %0.2fs\n', etime(clock, t0));
    
elseif choice == 1
    
    delta = kf_hyparr(A,B,H,K);
    
else
    
    error('unkown choice in hyparr')
    
end


% --------------------------------------------------------------------

function D = reorder(D)

expo = []; for i=0:size(D,1)-1, expo(end+1) = 2^i; end; expo=expo';
E = kron( ones(1,size(D,2)), expo );
[dummy, sortedIndices] = sort( sum(D.*E,1) );
D = D(:,sortedIndices);



% --------------------------------------------------------------------

% Buck's formula
n = 20;    % # hyperplanes
d = 2;      % dimension
N = 0;
for i=1:d
    N = N + nchoosek(n,i);
end;


