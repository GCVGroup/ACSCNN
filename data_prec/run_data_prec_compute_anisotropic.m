DATA_ROOT_DIR=fullfile(pwd,'..','FAUST');
SHAPE_DIR=fullfile(DATA_ROOT_DIR,'shapes');
NORM_LAPLACIAN_DIR=fullfile(DATA_ROOT_DIR,'aniso_norm_laplacian');


warning off;
mkdir(NORM_LAPLACIAN_DIR);
warning off;

SHAPES=dir(fullfile(SHAPE_DIR,'*.mat'));
SHAPES={SHAPES.name}';


% setting of anisotropic laplacian
% default n_angles=8, alpha=10
OPTIONS.ALPHA=10;
N_ANGLES=8;
OPTIONS.ANGLES=linspace(0,pi,N_ANGLES+1);
OPTIONS.ANGLES=OPTIONS.ANGLES(1:end-1);
OPTIONS.CURV_SMOOTH=10;

option.alpha=OPTIONS.ALPHA;
option.angle=0;
option.curv_smooth=OPTIONS.CURV_SMOOTH;

for s=1:numel(SHAPES)
    shapename=SHAPES{s};
    fprintf(1, '  %-30s \t ', shapename);
    time_start = tic;
    
    load(fullfile(SHAPE_DIR,shapename),'shape');
    
    Ls_norm=cell(1,numel(OPTIONS.ANGLES));
    
    for k=1:numel(OPTIONS.ANGLES)
        option.angle=OPTIONS.ANGLES(k);
        [W,A]=calc_anisotropic_laplacian([shape.X,shape.Y,shape.Z],shape.TRIV,option);
        Ls_norm{k}=shift_norm_laplacian(W,A);
    end
    
    L=diag_sparse_matrixs(Ls_norm);
    save(fullfile(NORM_LAPLACIAN_DIR,shapename),'L','-v7.3');
    
    % elasped time
    elapsed_time = toc(time_start);
    fprintf('%3.2fs\n',elapsed_time);
end