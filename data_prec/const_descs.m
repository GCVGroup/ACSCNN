DATA_ROOT_DIR=fullfile(pwd,'..','FAUST');
SHAPE_DIR=fullfile(DATA_ROOT_DIR,'shapes');
DESCS_DIR=fullfile(DATA_ROOT_DIR,'descs','const');


warning off;
mkdir(DESCS_DIR);
warning off;

SHAPES=dir(fullfile(SHAPE_DIR,'*.mat'));
SHAPES={SHAPES.name}';

for s=1:numel(SHAPES)
    shapename=SHAPES{s};
    fprintf(1, '  %-30s \t ', shapename);
    time_start = tic;
    
    load(fullfile(SHAPE_DIR,shapename),'shape');
    n=size(shape.X,1);
    desc=ones(n,1);
    
    save(fullfile(DESCS_DIR,shapename),'desc','-v7.3');
    
    % elasped time
    elapsed_time = toc(time_start);
    fprintf('%3.2fs\n',elapsed_time);
    
end