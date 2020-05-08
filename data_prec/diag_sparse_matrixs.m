function Ls_out=diag_sparse_matrixs(Ls)
% Ls 为cell，每个cell为一个n*n的稀疏矩阵
H=numel(Ls);
N=size(Ls{1},1);
ii=cell(H,1);
jj=cell(H,1);
vv=cell(H,1);

stride=0;

for s=1:H
    [i,j,v]=find(Ls{s});
    ii{s}=i+stride;
    jj{s}=j+stride;
    vv{s}=v;
    stride=stride+N;
end

ii=cell2mat(ii);
jj=cell2mat(jj);
vv=cell2mat(vv);

Ls_out=sparse(ii,jj,vv,N*H,N*H);
    
    