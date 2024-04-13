folder=dir("D:\MSc Books\Sem 4\Project\underwater_image_enhancement\images\raw");
files={folder.name};
files=files(3:end);  
for file=files
    im = im2double(imread(strcat("D:\MSc Books\Sem 4\Project\underwater_image_enhancement\images\raw\",file)));
    quality = CCF(im);
    disp(quality);
end