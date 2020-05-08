function [evecs, evals, W, A] = calc_ALB(vertices, faces, options)

n = size(vertices,1);
m = size(faces,1);

% Reorient mesh faces if they are inconsistent

adjacency_matrix = sparse([faces(:,1); faces(:,2); faces(:,3)], ...
                         [faces(:,2); faces(:,3); faces(:,1)], ...
    	                 ones(3 * m, 1), ...
                         n, n, 3 * m);
if any(any(adjacency_matrix > 1))
    options.method = 'slow';
    warning('Inconsistent face orientation. The mesh will be reoriented.')
    faces = transpose(perform_faces_reorientation(vertices,faces,options));
end
clear adjacency_matrix

[Umin, Umax, D] = avg_diffusion_tensor(vertices, faces, options.alpha, options.curv_smooth, options.angle);

% Construct the (anisotropic) stiffness matrix

W = sparse(n,n);

angles = zeros(size(faces));
for i=1:3
    i1 = mod(i-1,3)+1;
    i2 = mod(i  ,3)+1;
    i3 = mod(i+1,3)+1;
    pp = vertices(faces(:,i2),:) - vertices(faces(:,i1),:);
    qq = vertices(faces(:,i3),:) - vertices(faces(:,i1),:);
    pp = pp ./ repmat( max(sqrt(sum(pp.^2,2)),eps), [1 3] );
    qq = qq ./ repmat( max(sqrt(sum(qq.^2,2)),eps), [1 3] );
    angles(:,i1) = acos(sum(pp.*qq,2));
end

% For efficiency reasons we "iterate" over triangles. For each triangle, we
% consider its three edges and compute the stiffness values of those.

for i=1:3
    
    i1 = mod(i-1,3)+1;
    i2 = mod(i  ,3)+1;
    i3 = mod(i+1,3)+1;
    
    % Here we are considering the edge connecting i1 to i2.
    % In particular, angles(:,i3) is the angle opposing this edge.
    
    e1 = vertices(faces(:,i3),:)-vertices(faces(:,i2),:);
    e2 = vertices(faces(:,i1),:)-vertices(faces(:,i3),:);
    e1 = e1./repmat(sqrt(sum(e1.^2,2)),[1 3]);
    e2 = e2./repmat(sqrt(sum(e2.^2,2)),[1 3]);
    
    % Off-diagonal values.
    % Umin, Umax are the normalized directions of principal curvature.
    
    factore = -(1/2)*(...
        D(:,1).*(sum(e1.*Umin,2)).*(sum(e2.*Umin,2)) +...
        D(:,2).*(sum(e1.*Umax,2)).*(sum(e2.*Umax,2)))...
        ./sin(angles(:,i3));
    
    % diagonal factor
    factord = -(1/2)*(...
        D(:,1).*(sum(e1.*Umin,2).^2) + ...
        D(:,2).*(sum(e1.*Umax,2).^2)).*(cot(angles(:,i2))+cot(angles(:,i3)));
    
    % this explicitly makes W symmetric
    W = W + sparse(...
        [faces(:,i1); faces(:,i2); faces(:,i1)],...
        [faces(:,i2); faces(:,i1); faces(:,i1)],...
        [factore; factore; factord],...
        n, n);
end

% Construct the mass matrix

S_tri = zeros(m,1);
for k=1:m
    e1 = vertices(faces(k,3),:) - vertices(faces(k,1),:);
    e2 = vertices(faces(k,2),:) - vertices(faces(k,1),:);
    S_tri(k) = 0.5*norm(cross(e1,e2));
end
A = zeros(n,1);
for i=1:m
    A(faces(i,1)) = A(faces(i,1)) + S_tri(i)/3;
    A(faces(i,2)) = A(faces(i,2)) + S_tri(i)/3;
    A(faces(i,3)) = A(faces(i,3)) + S_tri(i)/3;
end
A = sparse(1:n,1:n,A);

[evecs, evals] = eigs(W, A, options.n_eigen, -1e-5, struct('disp', 0));

evals = diag(abs(real(evals)));
[evals, idx] = sort(evals);
evecs = evecs(:,idx);
evecs = real(evecs);

end
