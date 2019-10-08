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

### Task

The IPD calibration involves two alternating tasks, repeated some
number of times. In the first task, the subject will vertically align
a long horizontal line with the center of a static circle. In the
second, the subject will horizontally align a long vertical line with
the center of a static circle. Some more details (reminder text will
remind the subject of this before each run):

 - The subject presses `space` to start each task and to confirm that
   the line is centered.
 - The subject adjusts the line using the numpad keys (by default) or
   the arrows (if the `--arrow` flag is passed).
    - If numpad keys, there are two sets of keys for each direction,
      allowing both coarse and fine adjustment. `8`/`2` will do the
      coarse adjustment, `7`/`1` the fine, for vertical; `4`/`6` will
      do coarse, `1`/`3` the fine, for horizontal.
    - If arrows, we only do the coarse adjustment (and they do as
      you'd expect)
 - At any time, the subject can press `q` or `escape` to quit out of
   the task, which will quit without saving anything.

If the subject completes all runs, the outputs will be saved into a
csv file, `ipd_correction.csv`, saved in the output directory (by
default, on the user's Desktop within the `ipd` directory; this can be
changed with the `--output_dir` flag). If the file exists, we will
append the results to it, with the subject's name, their IPD, and an
entry for each run, containing the amount of vertical and horizontal
adjustment, in pixels and degrees, as well the monocular vergence
angle (which we compute), the fixation distance, screen width (in
pixels and cm), and a session number. See the [Using the
output](#using-the-output) section for details on how to use this csv.

### Usage

To use, run from the command line (from this directory):

```
python ipd_calibration.py subject_name binocular_ipd -s 1 2
```

where `subject_name` is the name of the subject, as you want it stored
for later retrieval, `binocular_ipd` is the subject's binocular IPD in
*CM* (not mm), and the integers after `-s` specify which screens to
run the task on (`1 2` is probably correct, but you may have to check
your machine's specific display set up to confirm; the default runs it
on screen `0`, which is only for testing purposes).

There are many optional arguments, all set via flags, most of which
you probably won't need to change. View the scripts help for more
information about the different required arguments and options (by
running `python ipd_calibration.py -h`), but here's a brief overview:

 - `--output_dir`/`-o`: set the directory where we look for the
   `ipd_correction.csv` file where we save output.
 - A variety related to the physical setup of the experiment: fixation
   distance (`--fixation_distance`), size of the screen in pixels
   (`--size`/`-p`) and cm (`--monitor_cm_width`/`-c`). These are all
   set for the FancyPants haploscope, and will probably need to change
   for other setups.
 - `--num_runs`/`-n`: how many runs of the calibration we should do
   (one run contains one vertical and horizontal calibration)
 - `--no_flip`/`-f`: by default, we flip everything horizontally since
   we'll be displaying this on a haploscope, which subjects view
   through a mirror. If your setup does not have a mirror between the
   subject and the screen (e.g., you're testing it on your laptop),
   pass this argument to remove the horizontal flip.
 - `--allow_large_ipd`: since we often talk about IPD in mm, but need
   them in cm for this script, the script will raise an exception by
   default if you pass an IPD larger than 10. If you actually do have
   an IPD larger than 10, passing this argument will suppress the
   exception and allow the script to continue.
 - Variety related to size of the stimuli: line length
   (`--line_length`/`-l`), width (for both line and circle
   `--line_width`/`-w`), and circle radius (`--circle_radius`/`-r`).
 - Arguments to control the timing of the line's blinking: the amount
   of time on (`--line_on_duration`/`-on`) and off
   (`--line_off_duration`/`-off`), both in seconds. The longer the on
   duration, the more likely subjects are to see the line "chasing"
   the circle, so you can play around with these to make sure it's not
   too big of a problem.
 - `--arrows`: by default, the numpad controls the location of the
   line, so we can allow for both coarse and fine adjustment. Passing
   this argument uses the arrows instead, which only allow for
   coarse. Use this if the keyboard you're using doesn't have a numpad
   (e.g., while testing on a laptop) or if you're using the `glfw`
   backend (see below).
 - `--win_type`: set the backend for the PsychoPy window. By default,
   we use `pyglet`, but I've occasionally had issues with a weird
   [`XF86VidModeGetGammaRamp failed`
   error](https://github.com/psychopy/psychopy/issues/2061). If you
   get that error and are unable to fix it, switching to the `glfw`
   backend will probably work (if you followed the above install
   instructions, you'll have the requirements for both on your
   machine). However, as of the time of this writing, the `glfw`
   backend does not [recognize the
   numpad](https://github.com/psychopy/psychopy/issues/2639), so
   you'll need to also pass the `--arrows` argument or the subject
   will not be able to move the line.

### Using the output

The output of the calibration task will be an ever-expanding
`ipd_correction.csv` file with all the information necessary you need
to perform the IPD correction. However, if you're not familiar with
interacting with this kind of information in python (using the
fantastic [pandas](https://pandas.pydata.org/) library), this won't be
that helpful to you.

In order to get the specific numbers you need, I've provided a quick
function, `csv_to_binocular_offset`, which provides the numbers most
people want: the horizontal and vertical offset for a given subject,
in either pixels or degrees, averaged across all runs they've
done. This is used by the `experiment.py` script to calculate the
proper offset.

# References

- Freeman, J., & Simoncelli, E. P. (2011). Metamers of the ventral
  stream. Nature Neuroscience, 14(9),
  1195–1201. http://dx.doi.org/10.1038/nn.2889
