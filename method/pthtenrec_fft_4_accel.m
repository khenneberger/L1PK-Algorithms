function out = pthtenrec_fft_4_accel(A,Y,para)
% Tensor Kaczmarz based high-order tensor recovery 
% pth power of l1 norm
% Linear transform is Fast Fourier Transform
% Inputs:
% Y: n1 x n2 x n3 ... x nm
% A: n1 x k x n3 ...x nm
% 
% para.alpha-step size for coordinate descent
%     .lambda-soft thresholding parameter
%     .maxit-maximum number of iterations
%     .bs-batch size
%     .control: 'rand', 'cyc'
%     .controltype: 'batch','block'
%
% Output:
% X: recovered tensor of size k x n2 x n3 x...xnm
%


ndim = length(size(Y));
nway = zeros(1,ndim);
for i = 1:ndim
    nway(i) = size(Y,i);
end
old_gamma  = 0;
old_lam = 0;

lambda = para.lambda;
bs     = para.bs;
p      = para.p;
alpha  = para.alpha;
nb = floor(nway(1)/bs); % number of blocks
display = true; % print iter number
numblock = para.block;
edges = round(linspace(1,nway(1)+1,numblock+1));


idx = randperm(nway(1));

k = size(A,2);
Xsize = nway;
Xsize(1) = k;
X = zeros(Xsize);
old_Z = X;
old_Zhat = X;
if isfield(para,'gth')
    err = zeros(para.maxit,1);
end

%out.Xhist = zeros(k,nway(2),nway(3),nway(4),para.maxit);
timer = zeros(para.maxit,1);
tic;

for i = 1:para.maxit
    if ~mod(i,10) && display && i>1
        fprintf('iter = %d\n',i);
        %fprintf('eta = %d\n',eta)
        %fprintf('lam = %d\n',lam);
        %fprintf('gamma = %d\n',gamma)
        fprintf('error = %d\n',err(i-1))
    end
    switch para.controltype
        case 'batch'
            switch para.control
                case 'rand'
                    % randomized batch
                    ii = ceil(rand*nb);
                    out.ii = ii;
                    ik = idx((bs*(ii-1)+1):(bs*ii));
                case 'cyc'
                    % cyclic batch
                    ii = i;
                    ik = mod((bs*(ii-1)+1):(bs*ii),nway(1));
                    ik(ik==0) = nway(1);
            end
        case 'block'
            switch para.control
                case 'rand'
                    block_select = randi(numblock);
                    ik           = edges(block_select):edges(block_select+1)-1;
                case 'cyc'
                  
                        block_select =  i;
                        block_select = mod(block_select,numblock);
                        if block_select ==0
                            block_select = numblock;
                        end
                        %disp(block_select);
                        ik           = edges(block_select):edges(block_select+1)-1;
            
            end
    end

   % normA = norm(A(ik,:,:,:),'fro')^2;
    normA = sum(A(ik,:,:,:).^2,"all");
    
    % adaptively update stepsize
    gamma  = (1+sqrt(1+4*(old_gamma)^2))/(2);
    eta = (1-old_gamma)/(gamma);
    old_gamma = gamma;

    % coordinate descent
    Zhat = old_Z + alpha*htprod_fft(htran(A(ik,:,:,:),'fft'),(Y(ik,:,:,:) ...
        -htprod_fft(A(ik,:,:,:),X))./normA); 
    Z = ((1-eta)*Zhat)+(eta*old_Zhat);
    old_Zhat = Zhat;
    old_Z = Z;

            % % Nesterov method (Bubeck)
            % lam = (1+sqrt(1+4*(old_lam)^2))/(2);
            % lam_next = (1+sqrt(1+4*(lam)^2))/(2);
            % old_lam = lam;
            % gamma = (1-lam)/lam_next;
            % 
            % Zhat = old_Z + eta*htprod_fft(htran(A(ik,:,:,:),'fft'),(Y(ik,:,:,:) ...
            % -htprod_fft(A(ik,:,:,:),X))./normA); 
            % Z = ((1-gamma)*Zhat)+(gamma*old_Zhat);
            % old_Zhat = Zhat;
            % old_Z= Z;

    % proximal operator
    if sum(sum(sum(isnan(Z))))>1
        break
    end 

    %% prox elementwise
    switch para.type
        case 'sparse'
            X = prox_pthv5(Z,lambda,p);
            
        case 'lowrank'
            X = prox_pth_singv5(Z,lambda,p);
    end
    
    
    if isfield(para,'gth') % relative error
        err(i) = norm(para.gth-X,'fro')./norm(para.gth,'fro');
        %disp(err(i));

    end
    if isfield(para,'tol')
            if err(i)<para.tol
                break
            end
    end
    timer(i) = toc;
    
end

out.finaliter = i;
out.X = X;
if isfield(para,'gth')
    out.err = err(1:i);
end

out.time = timer;
