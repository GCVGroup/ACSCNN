function [centroids,normals_tri] = calc_centroids(M)

centroids = (M.VERT(M.TRIV(:,1),:)+M.VERT(M.TRIV(:,2),:)+M.VERT(M.TRIV(:,3),:))./3;
normals_tri = cross(M.VERT(M.TRIV(:,2),:)-M.VERT(M.TRIV(:,1),:), M.VERT(M.TRIV(:,3),:)-M.VERT(M.TRIV(:,1),:));
normals_tri = normals_tri ./ repmat(sqrt(sum(normals_tri.^2,2)),1,3);

end
