function L=shift_laplacian(W,A)
n=size(W,1);
lmax=eigs(W,A,1);
W_=2*W./lmax-A;
area=full(diag(A));
L=sparse(1:n,1:n,1./area)*W_;

