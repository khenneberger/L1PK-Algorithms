function out = pthtenrec_fft_4(A,Y,para)
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

lambda = para.lambda;
alpha  = para.alpha;
bs     = para.bs;
p      = para.p;
nb = floor(nway(1)/bs); % number of blocks
display = true; % print iter number
numblock = para.block;
edges = round(linspace(1,nway(1)+1,numblock+1));


idx = randperm(nway(1));

k = size(A,2);
Xsize = nway;
Xsize(1) = k;
X = zeros(Xsize);
Z = X;

if isfield(para,'gth')
    err = zeros(para.maxit,1);
end

%out.Xhist = zeros(k,nway(2),nway(3),nway(4),para.maxit);



timer = zeros(para.maxit,1);
tic;
for i = 1:para.maxit
    if ~mod(i,2) && display
        fprintf('iter = %d\n',i);
        fprintf('err = %d\n', err(i-1));
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
    
    % coordinate descent
    Z = Z + alpha*htprod_fft(htran(A(ik,:,:,:),'fft'),(Y(ik,:,:,:) ...
        -htprod_fft(A(ik,:,:,:),X))./normA); 
    out.Z = Z;
    % proximal operator
    if sum(sum(sum(isnan(Z))))>1
        break
    end 
    lastX = X;

    %% prox elementwise
    switch para.type
        case 'sparse'
            X = prox_pthv5(Z,lambda,p);
            % [n1, n2,n3,n4]= size(Z);
            % for i1 = 1:n1
            %     for i2 = 1:n2
            %         for i3 = 1:n3
            %             for i4 = 1:n4
            %                 X(i1,i2,i3,i4) = prox_pthv2(Z(i1,i2,i3,i4),lambda,p);
            %             end
            %         end
            %     end
            % end
        case 'lowrank'
            X = prox_pth_singv5(Z,lambda,p);
    end
    
    if err(i)>100
       break
    end
    if isfield(para,'gth') % relative error
        err(i) = norm(para.gth(:)-X(:),'fro')/norm(para.gth(:),'fro');
        %disp(err(i));
        if isfield(para,'tol')
            if err(i)<para.tol
                break
            end
        end
    end

    timer(i) = toc;
    %disp(timer(i))
    
    
end

out.finaliter = i;
out.X = X;
if isfield(para,'gth')
    out.err = err(1:i);
end
out.time = timer;

