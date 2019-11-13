# foveated-metamers

Create metamers using models of the ventral stream and run experiments
to validate them

This project starts with a replication of Freeman and Simoncelli,
2011, out to higher eccentricities, and will extend it by looking at
spatial frequency information as well.

# Dockerfile

In order to build Dockerfile, have this directory and the most recent
version of `plenoptic` in the same directory and then FROM THAT
DIRECTORY (the one above this one), run `sudo docker build
--tag=foveated-metamers:YYYY-MM-dd -f foveated-metamers/Dockerfile
--compress .`. This ensures that we can copy plenoptic over into the
Docker container.

Once we get plenoptic up on pip (or even make it public on github), we
won't need to do this. At that time, make sure to replace
`foveated-metamers/environment.yml` with `environment.yml` and remove
the plenoptic bit.

Once image is built, save it to a gzipped tarball by the following:
`sudo docker save foveated-metamers:YYYY-MM-dd | gzip >
foveated-metamers_YYYY-MM-dd.tgz` and then copy to wherever you
need it.

# Requirements

This has only been tested on Linux, both Ubuntu 18.04 and
Fedora 29. It will probably work with minimal to no changes on OSX,
but there's no guarantee, and we definitely don't support Windows.

Need to make sure you have ffmpeg on your path when creating the
metamers, so make sure it's installed and on your path. When running
on NYU's prince cluster, can use `module load ffmpeg/intel/3.2.2` or,
if `module` isn't working (like when using the `fish` shell), just add
it to your path manually (e.g., on fish: `set -x PATH
/share/apps/ffmpeg/3.2.2/intel/bin $PATH`)

For demosaicing the raw images we use as inputs, you'll need to
install [dcraw](https://www.dechifro.org/dcraw/). If you're on Linux,
you can probably install it directly from your package manager. See
these [instructions](http://macappstore.org/dcraw/) for OSX. If you're
fine using the demosaiced `.tiff` files we provide, then you won't
need it.

Both provided conda environment files pin the versions of all the
python packages required to those used for the experiment. That's
probably not necessary, but is provided as a step to improve
reproducibility. We provide built Docker images for the same reason: 

## Experiment

## Experiment environment

For running the experiment, need to install `glfw` from your package
manager.

There are two separate python environments for this project: one for
running the experiment, and one for everything else. To install the
experimental environment, either follow [the minimal
install](#minimal-experiment-install) or do the following:

```
conda install -f environment-psychopy.yml
```

Then, to activate, run `conda activate psypy`.

## Environment everything else

To setup the environment for everything else:

```
conda install -f environment.yml
```

Then, to activate, run `conda activate metamers`.

The [plenoptic
library](https://github.com/LabForComputationalVision/plenoptic/) is
not yet on `pip`, so you'll have to download it manually (at that
link), then (in the `metamers` environment), navigate to that
directory and install it:

```
git clone git@github.com:LabForComputationalVision/plenoptic.git
cd plenoptic
pip install -e .
```

This environment contains the packages necessary to generate the
metamers, prepare for the experiment, and analyze the data, but it
*does not* contain the packages necessary to run the experiment. Most
importantly, it doesn't contain Psychopy, because I've found that
package can sometimes be a bit trickier to set up and is not necessary
for anything outside the experiment itself.

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
pip install psychopy==3.1.5 pyglet==1.3.2 numpy==1.17.0 h5py==2.9.0 glfw==1.8.2
```

And then if you're on Linux, fetch the wxPython wheel for you platform
from [here](https://extras.wxpython.org/wxPython4/extras/linux/gtk3/)
(for my setup, I used `wxPython-4.0.6-cp37-cp37m-linux_x86_64.whl`;
the `cp37` refers to python 3.7, I'm pretty sure, so that's very
important; not sure if the specific version of wxPython matters) and
install it with `pip install path/to/your/wxpython.whl`.

Everything should then hopefully work.

# Data dictionaries

Several pandas DataFrames are created during the course of this
experiment and saved as `.csv` files. In order to explain what the
different fields they have mean, I've put together some data
dictionaries, in the `data_dictionaries` directory. I tried to follow
[these
guidelines](https://help.osf.io/hc/en-us/articles/360019739054-How-to-Make-a-Data-Dictionary)
from the OSF. They are `.tsv` files and so can be viewed in Excel,
Google Sheets, a text editor, LibreOffice Calc, or loaded in to pandas
(`data_dict = pd.read_csv(data_dictionaries/metamer_summary.tsv,
'\t')`)

 - `metamer_summary.tsv`: during metamer synthesis, we save out a
   `summary.csv` file, which contains a DataFrame with one row,
   describing the metamer generated and some information about its
   synthesis. This data dictionary describes the columns in that
   DataFrame.
   
 - `all_metamer_summary.tsv`: in order to create the indices that
   determine the trias in the experiment, we gather together and
   concatenate all the `summary.csv` files and add one additional
   column, `image_name_for_expt`, then save the resulting DataFrame as
   `stimuli_description.csv`. This data dictionary describes that
   DataFrame's columns, which are identical to those in `summary.csv`,
   except for the addition.
   
- `experiment_df.tsv`: in order to analyze the data, we want to
  examine the images presented in each trial, what the correct answers
  was, and what button the subject pressed. We do this using
  `experiment_df.csv`, which we create for each experimental session
  (in a given session, one subject will see all trials for a given
  model; each subject, session pair has a different random seed used
  to generate the presentation index). Most of the DataFrame can be
  generated before the experiment is run (but after the index has been
  generated), but the final four columns (`subject_response,
  hit_or_miss, subject_name and session_number`) are only added when
  combining the subject's response information with the pre-existing
  `experiment_df`. We have two separate functions in `stimulus.py` for
  generating the DataFrame with and without subject response info, but
  we only save the completed version to disk.

# Usage

The general structure of the research project this repo describes is
as follows:

1. Develop models of the early visual system
2. Generate metamers for these models
3. Use psychophysics to set model parameters

The code for the models and general metamer synthesis are contained in
the [plenoptic
library](https://github.com/LabForComputationalVision/plenoptic/);
this repo has four main components: generate metamers (2), prepare for
the experiment (3), run the experiment (3), analyze the data from the
experiment (3), and run an IPD calibration (3; only necessary for
haploscope). How to use this repo for each of those tasks is described
below.

I use the [Snakemake](https://snakemake.readthedocs.io/en/stable/)
workflow management tool to handle most of the work involved in
generating the metamers, preparing for the experiment, and analyzing
the experiment output, so for everything except running the experiment
itself, you won't call the python scripts directly; instead you'll
tell `snakemake` the outputs you want, and it will figure out the
calls necessary, including all dependencies. This simplifies things
considerably, and means that (assuming you only want to run things,
not to change anything) you can focus on the arguments to `snakemake`,
which specify how to submit the jobs rather than making sure you get
all the arguments and everything correct.

## Setup

Make sure you've set up the software environment as described in the
[requirements](#requirements) section and activate the `metamers`
environment: `conda activate metamers`.

In order to generate these metamers in a reasonable time, you'll need
to have GPUs availabe. Without it, the code will not work; it could be
modified trivially by replacing the `gpu=1` with `gpu=0` in the
`get_all_metamers` function at the top of `Snakefile`, but generating
all the metamers would take far too much time to be
realistic. Additionally, PyTorch [does not
guarantee](https://pytorch.org/docs/stable/notes/randomness.html)
reproducible results between CPU and GPU executions, even with the
same seed, so you generating metamers on the CPU will not result in an
identical set of images, though (assuming the loss gets low enough),
they should still be valid metamers.

Decide where you want to place the metamers and data for this
project. For this README, it will be
`/home/billbrod/Desktop/metamers`. Edit the first line of the
`config.yml` file in this repo to contain this value (don't use the
tilde `~` for your home directory, python does not understand it, so
write out the full path).

Create that directory, download the tarball containing the reference
images and normalizing statistics, and unzip it into that directory:

```
mkdir /home/billbrod/Desktop/metamers
cd /home/billbrod/Desktop/metamers
wget -O- https://osf.io/5t4ju/download | tar xvz -C .
```

You should now have two directories here: `raw_images`, `ref_images`
and `norm_stats`. `raw_images` should contain four `.NEF` (Nikon's raw
format) images: `azulejos`, `flower`, `tiles`, and
`market`. `norm_stats` should contain a single `.pt` (pytorch) file:
`V1_cone-1.0_texture_degamma_cone_norm_stats.pt`. `ref_images` should
contain `einstein_size-256,256.png`, which we'll use for testing the
setup, as well as `.tiff` versions of the four raw images (the raw
images are provided in case you want to try a different demosaicing
algorithm than the one I did; if you're fine with that step, you can
ignore them and everything further will use the `.tiff` files found in
`ref_images`).

## Test setup

A quick snakemake rule is provided to test whether your setup is
working: `snakemake -j 4 -prk test_setup`. This will create a small number
of metamers, without running the optimization to completion. If this
runs without throwing any exceptions, your environment should be set
up correctly and you should have gpus available.

The output will end up in `~/Desktop/metamers/test_setup` and you can
delete this folder after you've finished.

## Generate metamers

Generating the metamers is very time consuming and requires a lot of
computing resources. We generate 108 images per model (4 reference
images * 3 seeds * 9 scaling values), and the amount of time/resources
required to create each image depends on the model and the scaling
value. The smaller the scaling value, the longer it will take and the
more memory it will require. For equivalent scaling values, V1
metamers require more memory and time than the RGC ones, but the RGC
metamers required for the experiment all have much smaller scaling
values. For the smallest of these, they require too much memory to fit
on a single GPU, and thus the length it takes increases drastically,
up to about 8 hours. For the V1 images, the max is about three hours.

The more GPUs you have available, the better.

If you wanted to generate all of your metamers at once, this is very
easy: simply running

```
snakemake -j n --resources gpu=n mem=m -prk --restart-times 3 --ri ~/Desktop/metamers_display/dummy_RGC_cone-1.0_gaussian_0_-1.txt ~/Desktop/metamers_display/dummy_V1_cone-1.0_norm_s6_gaussian_0_-1.txt
```

will do this (where you should replace both `n` with the number of
GPUs you have; this is how many jobs we run simultaneously; assuming
everything is working correctly, you could increase the `n` after `-j`
to be greater than the one after `--resources gpu=`, and snakemake
should be able to figure everything out, but I've had mixed success
with this; you should also replace `m` with the GB of RAM you have
available). `snakemake` will create the directed acyclic graph (DAG)
of jobs necessary to create those two txt files, which are just
placeholders that require all the metamers as their input.

However, you probably can't create all metamers at once on one
machine, because that would take too much time. You probably want to
split things up. If you've got a cluster system, you can configure
`snakemake` to work with it in a [straightforward
manner](https://snakemake.readthedocs.io/en/stable/executable.html#cluster-execution)
(snakemake also works with cloud services like AWS, kubernetes, but I
have no experience with that; you should google around to find info
for your specific job scheduler, see the small repo [I put
together](https://github.com/billbrod/snakemake-slurm) for using NYU's
SLURM system). In that case, you'll need to put together a
`cluster.json` file within this directory to tell snakemake how to
request GPUs, etc. Something like this should work for a SLURM system
(the different `key: value` pairs would probably need to be changed on
different systems, depending on how you request resources; the one
that's probably the most variable is the final line, gpus):

```
{
    "__default__":
    {
	"nodes": 1,
	"tasks_per_node": 1,
	"mem": "{resources.mem}GB",
	"time": "36:00:00",
	"job_name": "{rule}.{wildcards}",
	"cpus_per_task": 1,
	"output": "{log}",
	"error": "{log}",
	"gres": "gpu:{resources.gpus}"
    }
}
```

Every `create_metamers` job will use a certain number of gpus, as
given by `resources.gpu` for that job. In the snippet above, you can
see that we use it to determine how many gpus to request from the job
scheduler. On a local machine, `snakemake` will similarly use it to
make sure you don't run two jobs that require 3 gpus each if you only
have 4 gpus total, for example. Similarly, `resources.mem` provides an
estimate of how much memory (in GB) the job will use, which we use
similarly when requesting resources above. This is just an estimate
and, if you find yourself running out of RAM, you may need to increase
it in the `get_mem_estimate` function in `Snakefile.`

If you don't have a cluster available and instead have several
machines with GPUs so you can split up the jobs, that `dummy.txt` file
will help there as well: the structure of the file is
`dummy_{model_name}_{min_idx}_{max_idx}.txt`, where `{model_name}`
should be one of the two mentioned above, and `{min_idx}` and
`{max_idx}` are two integers from 0 to 108 inclusive (`max_idx` can
also be -1, which is equivalent to 108), where these index the list of
all 108 metamers generated for each model, returned in order of
increasing scaling value. Therefore, for each model, the lower
integers correspond to metamers that will take longer to generate, and
the higher ones correspond to metamers that will take less time (and
resources). You can use this to break up the generation of the
metamers in whatever way makes sense to you and your resources (note
because of how python indexing works, the different sets should have
overlapping `max_idx/min_idx` values; we go from
`metamers[min_idx:max_idx]`, so if you want to break this into e.g.,
two groups, you should create `dummy_{model_name}_0_64.txt` and
`dummy_{model_name}_64_108`). While messing with this, pay attention
to the `-j n` and `--resources gpu=n` flags, which tell snakemake how
many jobs to run at once and how many GPUs you have available,
respectively.

Note that I couldn't figure out any clever way to schedule jobs across
different GPUs, so the way I decided to handle it is to let jobs grab
the first GPU they think is available (available meaning that at most
30% of the its memory is being used, as determined by `GPUtil`). If
two jobs start at almost exactly the same time, one of them will
likely fail because it ran out of memory. To handle this, I recommend
adding the `--restart-times 3` flag to the snakemake call, as I do
above, which tells snakemake to try re-submitting a job up to 3 times
if it fails. Hopefully, the second time a job is submitted, it won't
have a similar problem. But it might require running the `snakemake`
command a small number of times in order to get everything
straightened out.

## Prepare for experiment

Once the metamers have all been generated, they'll need to be combined
into a numpy array for the displaying during the experiment, and the
presentation indices will need to generated for each subject.

For the experiment we performed, we had 30 subjects, with 3 sessions
per model. In order to re-generate the indices we used, you can simply
run `snakemake -prk gen_all_idx`. This will generate the indices for
each subject, each session, each model, as well as the stimuli array
(we actually use the same index for each model for a given subject and
session number; they're generated using the same seed).

This can be run on your local machine, as it won't take too much time
or memory.

The stimuli arrays will be located at:
`~/Desktop/metamers/stimuli/{model_name}/stimuli.npy` and the
presentation indices will be at
`~/Desktop/metamers/stimuli/{model_name}/{subject}_idx_sess-{num}.npy`. There
will also be a pandas DataFrame, saved as a csv, at
`~/Desktop/metamers/stimuli/{model_name}/stimuli_description.csv`,
which contains information about the metamers and their
optimization. It's used to generate the presentation indices as well
as to analyze the data.

You can generate your own, novel presentation indices by running
`snakemake -prk
~/Desktop/metamers/stimuli/{model_name}/{subject}_idx_sess-{num}.npy`,
replacing `{model_name}` with one of `'RGC_cone-1.0_gaussian',
'V1_cone-1.0_norm_s6_gaussian'`, `{subject}` must be of the format
`sub-##`, where `##` is some integer (ideally zero-padded, but this
isn't required), and `{num}` must also be an integer (this is because
we use the number in the subject name and session number to determine
the seed for randomizing the presentation order; if you'd like to
change this, see the snakemake rule `generate_experiment_idx`, and how
the parameter `seed` is determined; as long as you modify this so that
each subject/session combination gets a unique seed, everything should
be fine).

### Demo / test experiment

If you want to put together a quick demo, either to show someone what
the experiment looks like or to teach someone how to run the
experiment, a rule is provided for this. If you run `snakemake -prk
~/Desktop/metamers/stimuli/RGC_demo/sub-00_idx_sess-00.npy`, we'll
create a small stimulus array (it contains one image per scaling value
of the `azulejos` reference image, all with random seed 0) and the
indices necessary to present them. You can then follow the
instructions in the following section to run the experiment, using
`model_name=RGC_demo`, `subject=sub-00`, and `num=0`.

## Run experiment

To run the experiment, make sure that the stimuli array and
presentation indices have been generated and are at the appropriate
path. If you're running this experiment in a binocular setup (e.g., a
haploscope), make sure you've run the [IPD
calibration](#ipd-calibration) for the subject you're about to
run. It's recommended that you use a bite bar or eye-tracker or some
other way to guarantee that your subject remains fixated on the center
of the image; the results of the experiment rely very heavily on the
subject's and model's foveations being identical.

We want 3 sessions per subject per model. Each session will contain
all trials in the experiment, the only thing that differs is the
presentation order. Each trial lasts 4.1 seconds and is structured
like so:

```
|Image 1 | Blank  |Image 2 |  Blank  |Image X |  Blank  |
|--------|--------|--------|---------|--------|---------|
|200 msec|500 msec|200 msec|1000 msec|200 msec|2000 msec|
```

Image 1 and image 2 will always be two different images. They will
either be two metamers generated from the same reference image with
the same scaling value (and model), but different seeds, or one of
those images and the reference image it was generated from. Image X
will always be a repeat of either image 1 or 2, and the subject's job
is to press either 1 or 2 on the keyboard, in order to indicate which
image they think was repeated.

To run the experiment:

- Activate the `psypy` environment: `conda activate psypy`
- Start the experiment script from the command line: `python
   foveated_metamers/experiment.py ~/Desktop/metamers/stimuli/{model
   name}/stimuli.npy {subject} {num}`, where `{model_name}, {subject},
   and {num}` are as above
   - There are several other arguments the experiment script can take,
     run `python foveated_metamers/experiment.py -h` to see them, and
     see the [other arguments](#other-arguments) section for more
     information.
- Explain the task to the subject, as seen in the "say this to subject
  for experiment" section
- When the subject is ready, press the space bar to begin the task.
- You can press the space bar at any point to pause it, but the pause
  won't happen until the end of the current trial, so don't press it a
  bunch of times because it doesn't seem to be working. However, try
  not to pause the experiment at all.
- You can press q/esc to quit, but don't do this unless truly
  necessary.
- There will be a break half-way through the block. The subject can
  get up, walk, and stretch during this period, but remind them to
  take no more than 5 minutes. When they're ready to begin again,
  press the space bar to resume.
- The data will be saved on every trial, so if you do need to quit
  out, all is not lost. You can run the same command as above, and the
  experiment will pick up where you stopped.

Recommended explanation to subjects:

> In this experiment, you'll be asked to complete what we call an "ABX
> task": you'll view three images in sequence; the first two will
> always be different from each other, and the third will be a repeat
> of either the first or second. After the third image finishes
> displaying, you'll be prompted to answer whether the third image was
> a repeat of the first or second image; press either the 1 or 2
> button on the keyboard. All the images will be presented for a very
> brief period of time, so pay attention. Sometimes, the two images
> will be very similar; sometimes, they'll be very different. For the
> very similar images, we expect the task to be hard. Just do your
> best!

> For this experiment, fixate on the center of the image the whole
> time and try not to move your eyes.

> The task will last for about an hour, but there will be a break
> halfway through. During the break, you can move away from the
> device, walk around, and stretch, but please don't take more than 5
> minutes. Tell me when you're ready to begin again.

### Other arguments

The `experiment.py` takes several optional arguments, several of which
are probably relevant in order to re-run this on a different
experiment set up:

- `--screen` / `-s`: one or two integers which indicate which screens
  to use. If two numbers are passed, we'll run the experiment in
  binocular mode; if one is passed, we'll run it in monocular mode.
- `--no_flip` / `-f`: by default, the script is meant to be run on a
  haploscope, which means that all text is left-right flipped (because
  the subject is viewing the screens through a mirror). Passing this
  flag indicates that we should not flip the text.
- `--scren_size_pix` / `-p`: two integers which indicate the size of
  the screen(s) in pixels (if using two screens, they must have
  identical resolution).
- `--screen_size_deg` / `-d`: a single float which gives the length of
  the longest screen side in degrees (again, if using two screens,
  this must be identical for them).

For more details on the other arguments, run `python
foveated_metamers/experiment.py -h` to see the full docstring.

NOTE: While the above options allow you to run the experiment on a
setup that has a different screen size (both in pixels and in degrees)
than the intended one, the metamers were created with this specific
set up in mind. Things should be approximately correct on a different
setup (in particular, double-check that images are cropped, not
stretched, when presented on a smaller monitor), but there's no
guarantee. If you run this experiment, with these stimuli, on a
different setup, my guess is that the psychophysical curves will look
different, but that their critical scaling values should match; that
is, there's no guarantee that all scaling values will give images that
will be equally confusable on different setups, but the maximum
scaling value that leads to 50% accuracy should be about the same.

## Analyze experiment output

*In progress*

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
