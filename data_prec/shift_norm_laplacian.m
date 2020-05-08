function L=shift_norm_laplacian(W,A)
n=size(W,1);
area=full(diag(A));
L=sparse(1:n,1:n,1./area)*W;
M=full(diag(L));
M1=sparse(1:n,1:n,1./sqrt(M));
W_norm=M1*W*M1;
W_norm=(W_norm+W_norm')/2.0;% keep sysmetric

L=shift_laplacian(W_norm,A);


% W=normalized_laplacian(W);
% lmax=eigs(W,1);
% W=2*W./lmax-speye(n);
% area=full(diag(A));
% L=sparse(1:n,1:n,1./sqrt(area))*W*sparse(1:n,1:n,sqrt(area));
