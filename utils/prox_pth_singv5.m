function X = prox_pth_singv5(Z, lambda, eps)

[U,S,V] = ht_svd_fft(Z);
S = prox_pthv5(S,lambda,eps);
X = htprod_fft(U,htprod_fft(S,htran(V,'fft')));
end
