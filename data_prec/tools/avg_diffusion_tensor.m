function [Umin, Umax, D] = avg_diffusion_tensor(vertices, faces, alpha, curvature_smoothing, angle)

    function y = psi(x,coef)
        y = (1./(1+repmat(coef,size(x)).*abs(x)));%.*(x<=0) +...
            %(1+repmat(coef,size(x)).*x).*(x>0);
    end

% Compute curvature and normals

curv_options.curvature_smoothing = curvature_smoothing;
curv_options.verb = 0;
[Umin, Umax, Cmin, Cmax, ~, ~, normals] = compute_curvature(vertices', faces', curv_options);
Umin = Umin';
Umax = Umax';
normals = normals';

% (Umin,Umax,normals) form local reference frames at vertices.
% Now we compute *per-face* bases in global coordinates (R^3).

[~, Vmax] = interpolate_basis(Umin, Umax, normals, faces);

Vmax = Vmax./repmat(sqrt(sum(Vmax.^2,2)),[1 3]);

% Build per-face orthonormal basis using triangle edges

edge1 = vertices(faces(:,2),:)-vertices(faces(:,1),:);
edge2 = vertices(faces(:,3),:)-vertices(faces(:,1),:);
edge1 = edge1./repmat(sqrt(sum(edge1.^2,2)),[1 3]);
edge2 = edge2./repmat(sqrt(sum(edge2.^2,2)),[1 3]);
edge2 = edge2 - repmat(sum(edge1.*edge2,2),[1 3]).*edge1;
edge2 = edge2./repmat(sqrt(sum(edge2.^2,2)),[1 3]);

N = cross(edge2,edge1);
N = N./repmat(sqrt(sum(N.^2,2)), 1, 3);

Umax = Vmax - N.*repmat(sum(Vmax.*N,2),1,3);
Umax = Umax./repmat(sqrt(sum(Umax.^2,2)),[1 3]);

Umin = cross(N,Umax);
Umin = Umin./repmat(sqrt(sum(Umin.^2,2)),[1 3]);

% rotate
ca = cos(angle);
sa = sin(angle);
for ii=1:size(N,1)
    u = N(ii,:)';
    Rot = ...
        ca*eye(3) + ...
        sa*[0 -u(3) u(2) ; u(3) 0 -u(1) ; -u(2) u(1) 0] + ...
        (1-ca)*(u*u');
    Umin(ii,:) = Umin(ii,:)*Rot;
    Umax(ii,:) = Umax(ii,:)*Rot;
end

% Construct 2x2 matrix D

Cminmean = (1/3)*(Cmin(faces(:,1))+Cmin(faces(:,2))+Cmin(faces(:,3)));
Cmaxmean = (1/3)*(Cmax(faces(:,1))+Cmax(faces(:,2))+Cmax(faces(:,3)));
D = zeros(size(faces,1),2);
D(:,1) = 1/(1+alpha);%(1./(1+repmat(alpha,size(Cminmean)).*1e1));%psi(Cminmean,alpha);
D(:,2) = 1;%(1./(1+repmat(alpha,size(Cmaxmean)).*abs(1e-2)));%psi(Cmaxmean,alpha);

end
