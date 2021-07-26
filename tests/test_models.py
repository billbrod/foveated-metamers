#!/usr/bin/env python3
import os.path as op
import sys
import torch
import matplotlib.pyplot as plt
import pytest
import plenoptic as po
sys.path.append(op.join(op.dirname(op.realpath(__file__)), '..',
                        'extra_packages'))
import plenoptic_part as pop
sys.path.append(op.join(op.dirname(op.realpath(__file__)), '..',))
import foveated_metamers as fov

DTYPE = torch.float32
DATA_DIR = op.join(op.dirname(op.realpath(__file__)), '..', 'extra_packages',
                   'data')


@pytest.fixture(scope='package')
def img():
    return po.load_images(op.join(DATA_DIR, 'nuts.pgm'))[..., :64, :64]


# any tests that just use the default num_scale and order args, can use this
# fixture
@pytest.fixture(scope='package')
def obs(img):
    return fov.ObserverModel(1, img.shape[-2:])


class TestPooledVentralStream(object):

    def test_rgc(self):
        im = po.load_images(op.join(DATA_DIR, 'nuts.pgm'))
        rgc = pop.PooledRGC(.5, im.shape[2:])
        rgc(im)
        _ = rgc.plot_window_widths('degrees')
        _ = rgc.plot_window_widths('degrees', jitter=0)
        _ = rgc.plot_window_widths('pixels')
        _ = rgc.plot_window_areas('degrees')
        _ = rgc.plot_window_areas('degrees')
        _ = rgc.plot_window_areas('pixels')
        fig = po.imshow(im)
        _ = rgc.plot_windows(fig.axes[0])
        rgc.plot_representation()
        rgc.plot_representation_image()
        fig, axes = plt.subplots(2, 1, figsize=(5, 12))
        rgc.plot_representation(ax=axes[1])
        rgc.plot_representation_image(ax=axes[0])
        plt.close('all')

    def test_rgc_2(self):
        im = po.load_images(op.join(DATA_DIR, 'nuts.pgm'))
        rgc = pop.PooledRGC(.5, im.shape[2:], transition_region_width=1)
        rgc(im)
        _ = rgc.plot_window_widths('degrees')
        _ = rgc.plot_window_widths('degrees', jitter=0)
        _ = rgc.plot_window_widths('pixels')
        _ = rgc.plot_window_areas('degrees')
        _ = rgc.plot_window_areas('degrees')
        _ = rgc.plot_window_areas('pixels')
        fig = po.imshow(im)
        _ = rgc.plot_windows(fig.axes[0])
        rgc.plot_representation()
        rgc.plot_representation_image()
        fig, axes = plt.subplots(2, 1, figsize=(5, 12))
        rgc.plot_representation(ax=axes[1])
        rgc.plot_representation_image(ax=axes[0])
        plt.close('all')

    def test_rgc_metamer(self):
        # literally just testing that it runs
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=DTYPE).unsqueeze(0).unsqueeze(0)
        rgc = pop.PooledRGC(.5, im.shape[2:])
        metamer = pop.Metamer(im, rgc)
        metamer.synthesize(max_iter=3)
        assert not torch.isnan(metamer.synthesized_signal).any(), "There's a NaN here!"

    def test_rgc_save_load(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # first time we cache the windows...
        rgc = pop.PooledRGC(.5, im.shape[2:], cache_dir=tmp_path)
        rgc(im)
        rgc.save_reduced(op.join(tmp_path, 'test_rgc_save_load.pt'))
        rgc_copy = pop.PooledRGC.load_reduced(op.join(tmp_path,
                                                           'test_rgc_save_load.pt'))
        if not len(rgc.PoolingWindows.angle_windows) == len(rgc_copy.PoolingWindows.angle_windows):
            raise Exception("Something went wrong saving and loading, the lists of angle windows"
                            " are not the same length!")
        if not len(rgc.PoolingWindows.ecc_windows) == len(rgc_copy.PoolingWindows.ecc_windows):
            raise Exception("Something went wrong saving and loading, the lists of ecc windows"
                            " are not the same length!")
        # we don't recreate everything, e.g., the representation, but windows is the most important
        for i in range(len(rgc.PoolingWindows.angle_windows)):
            if not rgc.PoolingWindows.angle_windows[i].allclose(rgc_copy.PoolingWindows.angle_windows[i]):
                raise Exception("Something went wrong saving and loading, the angle_windows %d are"
                                " not identical!" % i)
        for i in range(len(rgc.PoolingWindows.ecc_windows)):
            if not rgc.PoolingWindows.ecc_windows[i].allclose(rgc_copy.PoolingWindows.ecc_windows[i]):
                raise Exception("Something went wrong saving and loading, the ecc_windows %d are"
                                " not identical!" % i)
        # ...second time we load them
        rgc = pop.PooledRGC(.5, im.shape[2:], cache_dir=tmp_path)

    def test_v1(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=DTYPE).unsqueeze(0).unsqueeze(0)
        v1 = pop.PooledV1(.5, im.shape[2:])
        v1(im)
        _ = v1.plot_window_widths('pixels')
        _ = v1.plot_window_areas('pixels')
        for i in range(v1.num_scales):
            _ = v1.plot_window_widths('pixels', i)
            _ = v1.plot_window_areas('pixels', i)
        v1.plot_representation()
        v1.plot_representation_image()
        fig, axes = plt.subplots(2, 1, figsize=(22, 18),
                                 gridspec_kw={'height_ratios': [1, 2]})
        v1.plot_representation(ax=axes[0])
        v1.plot_representation_image(ax=axes[1])
        plt.close('all')

    def test_v1_norm(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=DTYPE).unsqueeze(0).unsqueeze(0)
        v1 = pop.PooledV1(.5, im.shape[2:])
        stats = pop.optim.generate_norm_stats(v1, DATA_DIR, img_shape=(256, 256))
        v1 = pop.PooledV1(.5, im.shape[2:], normalize_dict=stats)
        v1(im)
        _ = v1.plot_window_widths('pixels')
        _ = v1.plot_window_areas('pixels')
        for i in range(v1.num_scales):
            _ = v1.plot_window_widths('pixels', i)
            _ = v1.plot_window_areas('pixels', i)
        v1.plot_representation()
        v1.plot_representation_image()
        fig, axes = plt.subplots(2, 1, figsize=(22, 18),
                                 gridspec_kw={'height_ratios': [1, 2]})
        v1.plot_representation(ax=axes[0])
        v1.plot_representation_image(ax=axes[1])
        plt.close('all')

    def test_v1_2(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=DTYPE).unsqueeze(0).unsqueeze(0)
        v1 = pop.PooledV1(.5, im.shape[2:], transition_region_width=1)
        v1(im)
        _ = v1.plot_window_widths('pixels')
        _ = v1.plot_window_areas('pixels')
        for i in range(v1.num_scales):
            _ = v1.plot_window_widths('pixels', i)
            _ = v1.plot_window_areas('pixels', i)
        v1.plot_representation()
        v1.plot_representation_image()
        fig, axes = plt.subplots(2, 1, figsize=(22, 18),
                                 gridspec_kw={'height_ratios': [1, 2]})
        v1.plot_representation(ax=axes[0])
        v1.plot_representation_image(ax=axes[1])
        plt.close('all')

    def test_v1_mean_luminance(self):
        for fname in ['nuts', 'einstein']:
            im = plt.imread(op.join(DATA_DIR, fname+'.pgm'))
            im = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            v1 = pop.PooledV1(.5, im.shape[2:])
            v1_rep = v1(im)
            rgc = pop.PooledRGC(.5, im.shape[2:])
            rgc_rep = rgc(im)
            if not torch.allclose(rgc.representation['mean_luminance'], v1.mean_luminance):
                raise Exception("Somehow RGC and V1 mean luminance representations are not the "
                                "same for image %s!" % fname)
            if not torch.allclose(rgc_rep, v1_rep[..., -rgc_rep.shape[-1]:]):
                raise Exception("Somehow V1's representation does not have the mean luminance "
                                "in the location expected! for image %s!" % fname)

    def test_v1_save_load(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # first time we cache the windows...
        v1 = pop.PooledV1(.5, im.shape[2:], cache_dir=tmp_path)
        v1(im)
        v1.save_reduced(op.join(tmp_path, 'test_v1_save_load.pt'))
        v1_copy = pop.PooledV1.load_reduced(op.join(tmp_path,
                                                         'test_v1_save_load.pt'))
        if not len(v1.PoolingWindows.angle_windows) == len(v1_copy.PoolingWindows.angle_windows):
            raise Exception("Something went wrong saving and loading, the lists of angle windows"
                            " are not the same length!")
        if not len(v1.PoolingWindows.ecc_windows) == len(v1_copy.PoolingWindows.ecc_windows):
            raise Exception("Something went wrong saving and loading, the lists of ecc windows"
                            " are not the same length!")
        # we don't recreate everything, e.g., the representation, but windows is the most important
        for i in range(len(v1.PoolingWindows.angle_windows)):
            if not v1.PoolingWindows.angle_windows[i].allclose(v1_copy.PoolingWindows.angle_windows[i]):
                raise Exception("Something went wrong saving and loading, the angle_windows %d are"
                                " not identical!" % i)
        for i in range(len(v1.PoolingWindows.ecc_windows)):
            if not v1.PoolingWindows.ecc_windows[i].allclose(v1_copy.PoolingWindows.ecc_windows[i]):
                raise Exception("Something went wrong saving and loading, the ecc_windows %d are"
                                " not identical!" % i)
        # ...second time we load them
        v1 = pop.PooledV1(.5, im.shape[2:], cache_dir=tmp_path)

    @pytest.mark.parametrize('window', ['cosine', 'gaussian'])
    def test_v1_scales(self, window):
        im = po.load_images(op.join(DATA_DIR, 'einstein.pgm'))
        v1 = pop.PooledV1(1, im.shape[-2:], std_dev=1, window_type=window)
        lum_rep = v1(im, ['mean_luminance'])
        more_rep = v1(im, ['mean_luminance', 0])
        if lum_rep.numel() >= more_rep.numel():
            raise Exception("V1 not properly restricting output!")
        if any([(i, 0) in v1.representation.keys() for i in [1, 2, 3]]):
            raise Exception("Extra keys are showing up in v1.representation!")
        if lum_rep.numel() != v1(im, ['mean_luminance']).numel():
            raise Exception("V1 is not dropping unnecessary output!")

    def test_v1_metamer(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE).unsqueeze(0).unsqueeze(0)
        v1 = pop.PooledV1(.5, im.shape[2:])
        metamer = pop.Metamer(im, v1)
        metamer.synthesize(max_iter=3)


class TestObserverModel(object):

    @pytest.mark.parametrize('num_scales', [1, 3, 4])
    @pytest.mark.parametrize('order', [1, 3])
    def test_obs_basic(self, num_scales, order, img):
        obs = fov.ObserverModel(1, img.shape[-2:], num_scales=num_scales,
                                order=order)
        rep = obs(img)
        total_elts = 0
        for k, v in obs._n_windows.items():
            total_elts += v
        assert rep.shape == (*img.shape[:2], total_elts)
        rep = obs.output_to_representation(rep)
        n_parts = obs.num_scales * (obs.order+1) + 1
        assert len(rep.keys()) == n_parts

    def test_obs_scales(self):
        img = po.load_images(op.join(DATA_DIR, 'nuts.pgm'))
        obs = fov.ObserverModel(1, img.shape[-2:])
        reduced_rep = obs(img, ['mean_luminance', 0])
        reduced_rep = obs.output_to_representation(reduced_rep, ['mean_luminance', 0])
        assert len(reduced_rep.keys()) == obs.order+1 + 1
        reduced_rep = obs(img, ['mean_luminance'])
        reduced_rep = obs.output_to_representation(reduced_rep, ['mean_luminance'])
        assert len(reduced_rep.keys()) == 1

    @pytest.mark.parametrize('num_scales', [1, 4])
    @pytest.mark.parametrize('order', [1, 3])
    def test_obs_plot_rep(self, num_scales, order, img):
        obs = fov.ObserverModel(1, img.shape[-2:], num_scales=num_scales,
                                order=order)
        rep = obs(img)
        fig, axes = obs.plot_representation(rep)
        n_parts = obs.num_scales * (obs.order+1) + 1
        assert len(axes) == n_parts
        fig, axes = obs.plot_representation_image(rep)
        assert len(axes) == obs.num_scales + 1

    @pytest.mark.parametrize('num_scales', [1, 4])
    @pytest.mark.parametrize('order', [1, 3])
    def test_obs_plot_other(self, num_scales, order, img):
        obs = fov.ObserverModel(1, img.shape[-2:], num_scales=num_scales,
                                order=order)
        obs.plot_window_areas()
        obs.plot_window_widths()
        obs.plot_windows()

    @pytest.mark.parametrize('num_scales', [1, 4])
    @pytest.mark.parametrize('order', [1, 3])
    def test_obs_other(self, num_scales, order, img):
        obs = fov.ObserverModel(1, img.shape[-2:], num_scales=num_scales,
                                order=order)
        rep = obs(img)
        obs.summarize_representation(rep)
        obs.summarize_window_sizes()

    def test_obs_save_load(self, tmp_path, img, obs):
        rep = obs(img)
        obs.summarize_representation(rep)
        obs.summarize_window_sizes()
        obs.save_reduced(op.join(tmp_path, 'test_obs_save_load.pt'))
        obs_copy = fov.ObserverModel.load_reduced(op.join(tmp_path, 'test_obs_save_load.pt'))
        for i in range(len(obs.PoolingWindows.angle_windows)):
            if not obs.PoolingWindows.angle_windows[i].allclose(obs_copy.PoolingWindows.angle_windows[i]):
                raise Exception("Something went wrong saving and loading, the angle_windows %d are"
                                " not identical!" % i)
        for i in range(len(obs.PoolingWindows.ecc_windows)):
            if not obs.PoolingWindows.ecc_windows[i].allclose(obs_copy.PoolingWindows.ecc_windows[i]):
                raise Exception("Something went wrong saving and loading, the ecc_windows %d are"
                                " not identical!" % i)

    def test_obs_update_plot(self, img, obs):
        rep = obs(img)
        fig, axes = obs.plot_representation(rep)
        obs.update_plot(axes, torch.rand_like(rep))
        for ax, data in zip(axes, obs.output_to_representation(rep).values()):
            plotted_data = torch.tensor([s[1, 1] for s in
                                         ax.containers[0].stemlines.get_segments()])
            assert torch.any(data != plotted_data)
        obs.update_plot(axes, rep)
        for ax, data in zip(axes, obs.output_to_representation(rep).values()):
            plotted_data = torch.tensor([s[1, 1] for s in
                                         ax.containers[0].stemlines.get_segments()])
            assert torch.all(data == plotted_data)

    def test_obs_to(self, img, obs):
        # can't test float16, because some operations are unsupported
        obs.to(torch.float64)
        rep = obs(img)
        assert rep.dtype == torch.float64
        obs.plot_representation(rep)
        obs.to(torch.float32)
        rep = obs(img)
        assert rep.dtype == torch.float32
        obs.plot_representation(rep)

    def test_obs_metamer(self, img, obs):
        metamer = pop.Metamer(img, obs)
        metamer.synthesize(max_iter=3, coarse_to_fine='together')
