% uses https://github.com/irisgroen/LGNstatistics/, this currently hardcodes
% stuff and is not the right way to handle it, but it's temporary

addpath('~/Documents/LGNstatistics/CEandSCmatlab')

names = {'azulejos', 'tiles', 'bike', 'graffiti', 'llama', 'rooves', 'store', 'terraces', 'trunk_symmetric', 'treetop_symmetric', 'grooming_symmetric', 'palm_symmetric', 'ground_symmetric', 'leaves_symmetric', 'portrait_symmetric', 'troop_symmetric', 'termite_symmetric', 'digging_symmetric', 'plinth_symmetric', 'quad_symmetric', 'plaza_symmetric', 'penn_symmetric', 'dorm_symmetric', 'pillar_symmetric', 'highway_symmetric', 'statue_symmetric', 'graves', 'ivy', 'nyc', 'redrocks', 'redrocks2', 'rocks', 'terracotta', 'valley', 'boats', 'charlesbroadway', 'gnarled', 'house', 'lettuce', 'lodge', 'rowhouses'};
gamma_corrected = '_gamma-corrected';
% gamma_corrected = '';
path_template = '~/Desktop/metamers/ref_images_preproc/%s%s_range-.05,.95_size-2048,2600.png';
CE = nan(length(names),1);
SC = nan(length(names),1);
Beta = nan(length(names),1);
Gamma = nan(length(names),1);

% default dotpitch value
dotpitch = .35*(10^-3);
for ii = 1:length(names)
    name = names{ii}
    im = imread(sprintf(path_template, name, gamma_corrected));
    im = double(im) / 65535;

    [ce, sc, beta, gamma, par, imfovbeta, mag, imfovgamma] = LGNstatistics(im, .4, dotpitch, 40, 40);
    fig = figure('position', [0, 0, 3500, 800]);
    subplot(1,3,1);
    imshow(im);
    par = par./max(par(:));
    par = uint8(par*255);
    title(name)
    subplot(1,3,2);
    imshow(par);
    fov = nan(size(par));
    fov(imfovbeta)=255;
    hold on;
    imagesc(uint8(fov), 'AlphaData', .2);
    title(sprintf('Parvo, ce: %.03e, beta: %.03e', ce, beta));
    mag = mag ./ max(mag(:));
    mag = uint8(mag*255);
    subplot(1,3,3);imshow(mag);
    fov = nan(size(mag));
    fov(imfovgamma) = 255;
    hold on;
    imagesc(uint8(fov), 'AlphaData', .2);
    title(sprintf('Magno, sc: %.03e, gamma: %.03e', sc, gamma));
    saveas(fig, sprintf('~/Desktop/metamers/figures/image_select/%s%s_large_fov.png', name, gamma_corrected))
    CE(ii) = ce;
    SC(ii) = sc;
    Beta(ii) = beta;
    Gamma(ii) = gamma;
end

matname = sprintf('~/Desktop/metamers/figures/image_select/lgnstats%s_large_fov.mat', gamma_corrected);
try
    save(matname, 'CE', 'SC', 'Beta', 'Gamma', 'names')
catch
    delete(matname);
    save(matname, 'CE', 'SC', 'Beta', 'Gamma', 'names');
end
