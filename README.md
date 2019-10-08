# foveated-metamers

Create metamers using models of the ventral stream and run experiments to validate them

This project starts with a replication of Freeman and Simoncelli,
2011, out to higher eccentricities, and will extend it by looking at
spatial frequency information as well.

# Dockerfile

In order to build Dockerfile, have this directory and the most recent
version of `plenoptic` in the same directory and then FROM THAT
DIRECTORY (the one above this one), run `sudo docker build
--tag=foveated-metamers:YYYY-MM-dd -f foveated-metamers/Dockerfile
.`. This ensures that we can copy plenoptic over into the Docker
container.

Once we get plenoptic up on pip (or even make it public on github), we
won't need to do this. At that time, make sure to replace
`foveated-metamers/environment.yml` with `environment.yml` and remove
the plenoptic bit.

Once image is built, save it to a gzipped tarball by the following:
`sudo docker save foveated-metamers:YYYY-MM-dd | gzip >
foveated-metamers_YYYY-MM-dd.tar.gz` and then copy to wherever you
need it.

# Requirements

Need to make sure you have ffmpeg on your path when creating the
metamers, so make sure it's installed.

When running on NYU's prince cluster, can use `module load
ffmpeg/intel/3.2.2` or, if `module` isn't working (like when using the
`fish` shell), just add it to your path manually (e.g., on fish: `set
-x PATH /share/apps/ffmpeg/3.2.2/intel/bin $PATH`)

For running the experiment, need to install `glfw` from your package
manager.

## Source images

We use the
[LIVE-NFLX-II](http://live.ece.utexas.edu/research/LIVE_NFLX_II/live_nflx_plus.html)
data set to get some input images. We provide the images (which are
frames from some of these movies, with some processing: converting to
grayscale, cropped appropriately, and saved as pgm) in our
[ref_images](https://osf.io/5t4ju) tarball, but the link is presented
here if you wish to examine the rest of the data set.

We also use several images from pixabay. Similar to the LIVE-NFLX-II,
these images have had some processing done to them, but the links to
the originals are presented here:
[japan](https://pixabay.com/photos/japan-travel-nature-asia-plant-4141578/),
[sheep](https://pixabay.com/photos/sheep-agriculture-animals-17482/),
[street](https://pixabay.com/photos/street-people-road-many-3279061/),
[refuge](https://pixabay.com/photos/refuge-dolomites-fog-prato-steep-4457275/),
[trees](https://pixabay.com/photos/trees-fir-forest-nature-conifers-3410830/).


## Minimal experiment install

If you just want to run the experiment and you want to install the
minumum number of things possible, the following should allow you to
run this experiment. Create a new virtual environment and then:

```
pip install psychopy==3.1.5 pyglet==1.3.2 numpy h5py glfw
```

And then if you're on Linux, fetch the wxPython wheel for you platform
from [here](https://extras.wxpython.org/wxPython4/extras/linux/gtk3/)
and install it with `pip install path/to/your/wxpython.whl`.

Everything should then hopefully work.

## IPD calibration

We also include a script, `ipd_calibration.py`, which should work with
the `psypy` environment set up from `environment-psychopy.yml` or the
minimal experiment install above. This script is for use with a
haploscope in order to determine how far apart the centers of the
images in each eye should be to allow the subject to successfully fuse
them. If their position is constant, this shouldn't change for a given
subject (assuming your experimental setup doesn't change).

To use, run from the command line. View that scripts help for more
information about the different required arguments and options:
`python foveated_metamers/ipd_calibration.py -h`.

# References

- Freeman, J., & Simoncelli, E. P. (2011). Metamers of the ventral
  stream. Nature Neuroscience, 14(9),
  1195–1201. http://dx.doi.org/10.1038/nn.2889
