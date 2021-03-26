% used by Snakefile, need to have Jeremy Freeman's metamer code
% (https://github.com/freeman-lab/metamers/) and matlabPyrTools
% (https://github.com/LabForComputationalVision/matlabPyrTools) on your matlab
% path before running this

function freeman_windows(scale, savedir)

    oim = rand(512, 512);
    opts = metamerOpts(oim, sprintf('scale=%s', scale));
    masks = mkImMasks(opts);
    im = plotWindows(masks, opts);

    save(sprintf('%s/plotwindows.mat', savedir), 'im', '-v7.3')
    save(sprintf('%s/masks.mat', savedir), '-struct', 'masks', '-v7.3')

end
