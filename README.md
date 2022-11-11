# foveated-metamers

This repo contains the code for a vision science experiment investigating how
human perception changes across the visual field using behavioral experiments
and computational models inspired by the earlys tages of visual processing. We
use these models to investigate what people cannot see, an approach that has a
long history of vision science. If we know what information people are
insensitive to, we can discard it or randomize it, and the resulting image
should appear unchanged from the original.

See the [VSS 2020 poster](https://osf.io/aketq/) for scientific details. You may
also be interested in the
[website](https://users.flatironinstitute.org/~wbroderick/metamers/) we put
together for browsing through the synthesized images.

If you re-use some component of this project in an academic publication, see the
[citation](#citation) section for how to credit us.

# Usage

The data and code for this project are shared with the primary goal of enabling
reproduction of the results presented in the associated paper. Novel analyses
should be possible, but no guarantees.

To that end, we provide [several entrypoints into the data](#data) for
re-running the analysis, with a script to automate their download and proper
arrangement.

As a note: `snakemake` commands create files. I recommend adding `-n` to any
`snakemake` command when you run it for the first time. This will do a "dry
run", so you can see what steps `snakemake` will take, without running anything.

The following steps will walk you through downloading the fully-processed data
and recreating the figures, read further on in this README for details:
1. Clone this repo.
2. Open `config.yml` and modify the `DATA_DIR` path to wherever you wish to
   download the data (see [config.yml](#config.yml) section for details on this
   file).
3. Install the required software:
   - Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) on your
     system for python 3.7.
   - Install [mamba](https://github.com/mamba-org/mamba): `conda install mamba
     -n base -c conda-forge`
   - Navigate to this directory and run `mamba env create -f environment.yml` to
     install the environment.
   - Run `conda activate metamers` to activate the python environment.
   - Additionally, install [inkscape](https://inkscape.org/).
4. Run `python download_data.py synthesis_input mcmc_fits figure_input ` to
   download the data required to create the papers in the main figure (this is
   about 20GB).
5. Run `snakemake -k -j N reports/paper_figures/fig-XX.svg`
   (where `N` is the number of cores to use in parallel) to recreate a given
   figure from the paper (note the number must be 0-padded, i.e., `fig-01.svg`,
   *not* `fig-1.svg`). These will end up in the `reports/paper_figures/`
   directory. Note that they are svgs, a vector file format. If your default
   image viewer cannot open them, your browser can. They can be converted to
   pdfs using [inkscape](https://inkscape.org/) or Adobe Illustrator.
6. If you wish to create all the figures from the main body of the text, run
   `snakemake -k -j N main_paper_figures`. If one job fails, this
   will continue to run the others (that's what the `-k` flag means).

If you wish to create the figures in the appendix as well:
1. Download the additional data required: `python download_data.py stimuli
   freeman2011_check_output` (this is about 14GB).
2. Run `snakemake -k -j N reports/paper_figures/fig-AY-XX.svg` (where `XX` must
   again be 0-padded, but `Y` does not) to create figure `XX` from appendix `Y`
   or `snakemake -k -j N appendix_figures` to create all the figures from
   appendices 1 through 5.
3. The figures in appendix 6 have been split off because they require an
   additional 24GB data set, so to create these:
    - Download the additional data: `python download_data.py mcmc_compare`.
    - Create a single figure with `snakemake -k -j N
      reports/paper_figures/fig-A6-XX.svg` or all of them with `snakemake -k -j
      N appendix_figures_mcmc_compare`.
      
Some notes about the above:
1. The workflow for figure creation looks intimidating: because parallelization
   is easy, I split up the process into many small jobs. Therefore, there's ~100
   jobs for each of the above `main_paper_figures` and `appendix_figures`. Don't
   worry! They generally don't take that much time.
2. The overall workflow is very long (going back to preparing the images for
   metamer synthesis), and so sometimes `snakemake` can take a long time to
   determine what to run. This problem can get exacerbated if the file
   modification timestamps get thrown off, so that `snakemake` thinks it needs
   to re-create some of the existing files. To limit the search space and force
   `snakemake` to only consider figure-related rules, use the included
   `reports/figure_rules.txt` and the `--allowed-rules` flag: `cat
   reports/figure_rules.txt | xargs snakemake -prk main_paper_figures
   --allowed-rules`. You can pass any argument to `snakemake`, as long as the
   command ends with `--allowed-rules`.
2. Several of the figures in the paper (e.g., figure 4) include example metamers
   or other large images. We link these images into the figure, rather than
   embed them, until the very end, in order to reduce file size. Embedding them
   requires `inkscape` and an attached display (so it cannot be run on e.g., a
   compute cluster). You can do all the steps *except* embedding the large
   images by appending `_no_embed` to the file or target name. So, you would
   create `reports/paper_figures/fig-04_no_embed.svg` rather than
   `reports/paper_figures/fig-04.svg` to create that single figure, or call
   `snakemake -k -j N main_paper_figures_no_embed` / `snakemake -k -j N
   appendix_figures_no_embed` to create all of the main paper / appendix figures
   without embedding.
    - This allows you to run everything except the embedding on one machine that
      may be more powerful but lack a display (such as a compute cluster), and
      then finish up on e.g., your laptop. However, the paths used to link
      images will almost certainly *not work* when moving to a different
      machine, so if you view `fig-04_no_embed.svg`, you will see empty red
      squares where the images should go. When embedding the images in, we
      correct the paths, so this is not a problem.
    - It is possible that `snakemake` will get confused when you switch machines
      and decide that it wants to re-run steps because the file modification
      timestamps appear out of order (this might happen, in particular, because
      of `TEXTURE_DIR`, which is used at the very beginning of the workflow;
      point it to something old or non-existant to avoid this!). To prevent
      this, use the same trick as above: append `--allowed-rules
      embed_bitmaps_into_figures main_figures appendix_figures` to any
      `snakemake` command to ensure that it will only run the embedding rule.
   
Reproducing someone else's research code is hard and, in all likelihood, you'll
run into some problem. If that happens, please [open an
issue](https://github.com/billbrod/foveated-metamers/issues) on this repo, with
as much info about your machine and the steps you've taken as possible, and I'll
try to help you fix the problem.

To understand what the `snakemake` command is doing, see the [What's going
on?](https://github.com/billbrod/spatial-frequency-preferences#whats-going-on)
section I wrote in the readme for another project (here's the [zenodo
doi](https://zenodo.org/record/6028263) in case that disappears).

## What if I want to do more than recreate the figures?

I have focused on enabling others to recreate the figures, but you should be
able to use this repo to do everything in the paper. In particular, you might
want to:

### ... examine the metamers synthesized for this project

We've put together a
[website](https://users.flatironinstitute.org/~wbroderick/metamers/) where you
can browse all the metamers synthesized for this project, filtering and sorting
by their metadata.

If you'd like to bulk download all of them, you can do so from the [OSF
page](https://osf.io/67tbe/files/osfstorage), see its
[README](https://osf.io/kjf75) for how they're organized.

### ... synthesize some metamers 

I don't recommend using this repo to do this unless you're trying to do exactly
what I did (and even so, see [here](#notes-on-reproducibility)). If you want to
synthesize your own metamers, see
[plenoptic](https://github.com/LabForComputationalVision/plenoptic/) for a
better tested, better documented, and more general implementation of metamer
synthesis (plus more!).

But if you still want to try to synthesize some metamers using the code in this
repo, download the `figure_input` data set and look at the path of the
downloaded metamers. You can use snakemake to create metamers like that, and
most parts of the path are options related to metamer synthesis, see
`METAMER_TEMPLATE_PATH` in `config.yml`, as well as the `create_metamers` rule
in `Snakefile` to get a sense for what these are.

To recreate any of my metamers, you'll also need to download the
`synthesis_input` data set, which includes the target images we used, as well as
statistics used for normalizing the models' representation.

You should also be aware that the pooling windows are very large once you get
below `scaling=0.1`, so I would start with a larger window size. It is also strongly recommended to use a GPU, which will greatly speed up synthesis.

### ... see what the experiment was like

The OSF project contains a video of [a single training
run](https://osf.io/7vm43) shown to participants before performing the energy
model original vs. synthesized comparison task. In it, participants view the
metamers for two target images (`tiles` and `azulejos`) at the smallest and
largest scaling values for this comparison (`.063` and `.27`), comparing them
against the original image. Participants receive feedback in the training (the
central dot turns green when they answer correctly) and are told their
performance at the end of the run; no feedback was given in the actual
experiment. It was expected that participants would get close to 100% on the
easy trials (large scaling) and close to 50% on the hard trials (small scaling).

If you wish to try the experiment yourself, set up your environment for the
[experiment](#experiment-environment) and download the experiment training
tarball: `python download_data.py experiment_training`. You can then follow the
instructions in the [Training](#training) section of this readme (note that you
won't be able to use the `example_images.py` script; if you're interested in
this, open an issue and I'll rework it).

### ... run the full experiment

First, Set up your environment for the [experiment](#experiment-environment) and
download the stimuli: `python download_data.py stimuli`. 

You may also want to download the files used in the training and [try that
out](#...-see-what-the-experiment-was-like).

For a given model and comparison, the full expeirment consists of 3 sessions,
with 5 runs each. A single session lasts about an hour, with small breaks built
in between runs, each of which lasts about 10 minutes. Each session contains 5
target images, so that each subject sees 15 of the total 20. All subjects see
the first 10, then the final 10 are split into two groups, with even-numbered
subjects seeing the first group, odd-numbered the second.

You'll need to generate presentation indices (which define what order the images
are presented in; the ones for the training task are included in their tarball).
To do so, use snakemake: `snakemake -prk
{DATA_DIR}/stimuli/{model_name}/task-split_comp-{comp}/{subj_name}/{subj_name}_task-split_comp-{comp}_idx_sess-{sess_num}_run-{run_num}.npy`,
where:


- `{DATA_DIR}`: the `DATA_DIR` field from the `config.yml` file 
- `{model_name}`: either `RGC_norm_gaussian` (for the luminance model) or
  `V1_norm_s6_gaussian` (energy)
- `{comp}`: one of `met`, `ref`, `met-natural`, `ref-natural` or
  `ref-downsample-2`. This should match the `{comp}` wildcard from the stimulus
  file you downloaded.
- `{subj_name}`: has the form `sub-##`, where `##` a 0-padded integer. If this
  integer lies between 0 and 7 (inclusive), this will be the same presentation
  order as used in our experiment.
- `{sess_num}`: 0-padded itneger between 0 and 2 (inclusive). The session
  determines which set of 5 target images are included.
- `{run_num}`: 0-padded integer between 0 and 4 (inclusive). Each run contains 3
  target images, so that `run-01` contains target images `{A,B,C}`, `run-02`
  contains `{B,C,D}`, `run-03` contains `{C,D,E}`, `run-04` contains `{D,E,A}`,
  and `run-05` contains `{E,A,B}`.
  
You'll probably want to generate all the indices for a subject for a given model
and comparison at once. You can do that by generating the dummy file: `snakemake
-prk
{DATA_DIR}/stimuli/{model_name}/task-split_comp-{comp}/{subj_name}/{subj_name}_task-split_comp-{comp}_idx_tmp.txt`.

Then read the [Run experiment](#run-experiment) section of this readme.

### ... refit the psychophysical curves

# Setup

The analyses were all run on Linux (Ubuntu, Fedora, and CentOS, several
different releases). Everything should work on Macs. For Windows, I would
suggest looking into the [Windows Subsystem for
Linux](https://docs.microsoft.com/en-us/windows/wsl/about), as Windows is very
different from the others.

## Software requirements

Need to make sure you have ffmpeg on your path when creating the metamers, so
make sure it's installed and on your path. I have had a lot of trouble using
`module` to load the ffmpeg present on NYU prince, and so recommend installing
[a static build](https://www.johnvansickle.com/ffmpeg/faq/) and using that
directly (note that I have not had this problem with NYU greene or Flatiron
Institute's rusty, so it appears to be cluster-specific).

Both provided conda environment files pin the versions of all the
python packages required to those used for the experiment. That's
probably not necessary, but is provided as a step to improve
reproducibility.

If you're using GPUs to create images, you'll also need `dotlockfile`
on your machine in order to create the lockfiles we use to prevent
multiple jobs using the same GPU.

Other requirements:
- inkscape, at least version 1.0.2 (used for figure creation).

## Experiment environment

For running the experiment, need to install `glfw` from your package
manager.

There are two separate python environments for this project: one for
running the experiment, and one for everything else. To install the
experimental environment, either follow [the minimal
install](#minimal-experiment-install) or do the following:

```
conda env create -f environment-psychopy.yml
```

Then, to activate, run `conda activate psypy`.

PsychoPy provides multiple backends. I'm now using the `pyglet` backend, but
I've occasionally had issues with a weird [`XF86VidModeGetGammaRamp failed`
error](https://github.com/psychopy/psychopy/issues/2061). If you get that error
and are unable to fix it, switching to the `glfw` backend will probably work (if
you followed the above install instructions, you'll have the requirements for
both on your machine). I've also had issues with `glfw` where it doesn't record
the key presses before the pause and run end during the experiment, which means
those trials aren't counted and may mess up how `analysis.summarize_trials`
determines which keypress corresponds to which trial. If you switch to `glfw`,
you should carefully check that.

## Environment everything else

To setup the environment for everything else:

```
git submodule sync
git submodule update --init --recursive
conda env create -f environment.yml
```

Then, to activate, run `conda activate metamers`.

This environment contains the packages necessary to generate the
metamers, prepare for the experiment, and analyze the data, but it
*does not* contain the packages necessary to run the experiment. Most
importantly, it doesn't contain Psychopy, because I've found that
package can sometimes be a bit trickier to set up and is not necessary
for anything outside the experiment itself.

## Source images

We use images from the authors' personal collection and the [UPenn Natural Image
Database](http://tofu.psych.upenn.edu/~upennidb/) as the targets for our metamer
generation. This is because we need images that are large, linear (i.e., their
pixel intensities are proportional to photon count, as you get from an image
that has not been processed in any way), and openly-licensed. See the
[Setup](#setup) section for details on how to obtain the images from the Open
Science Foundation website for this project, along with the statistics used to
normalize the V1 model and a small image of Albert Einstein for testing the
setup.

Authors' personal collection: 
- WFB: azulejos, tiles, bike, graffiti, llama, terraces
- EPS: ivy, nyc, rocks, boats, gnarled, lettuce

UPenn Natural Image Database: treetop (cd01A/DSC_0033), grooming
(cd02A/DSC_0011), palm (cd02A/DSC_0043), leaves (cd12A/DSC_0030), portrait
(cd58A/DSC_0001), troop (cd58A/DSC_0008).

Unpublished photos from David Brainard: quad (EXPOSURE_ASC/DSC_0014), highway
(SNAPSHOTS/DSC_0200).

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

# Data

# Directory structure

 - `Snakefile`: used by snakemake to determine how to create the files for this
   project. Handles everything except the experiment.
 - `foveated_metamers/`: library of functions used in this project
    - `create_metamers.py`: creates metamers.
    - `stimuli.py`: assembles the various metamer images into format required
      for running the experiment.
    - `distances.py`: finds distance in model space between images in an
      efficient way.
    - `experiment.py`: runs experiment.
    - `analysis.py`: basic analyses of behavioral data (gets raw behavioral data
      into format that can fit by psychophysical curves).
    - `curve_fit.py`: fits psychophysical curves to real or simulated data using
      `pytorch`. We didn't end up using this method of fitting the curves.
    - `simulate.py`: simulate behavioral data, for checking `curve_fit.py`
      performance, as well as how many trials are required.
    - `mcmc.py`: use Markov Chain Monte Carlo (MCMC) to fit a probabilistic
      model of psychophysical curves with `numpyro`. This is how the curves
      presented in the paper were fit.
    - `statistics.py`: compute some other image statistics (heterogeneity,
      Fourier amplitude spectra, etc).
    - `plotting.py`: plotting functions.
    - `figures.py`: creates various figures.
    - `compose_figures.py`: combines plots (as created by functions in
      `figures.py`) into multi-panel figures.
    - `other_data.py`: functions to fit a line (hinged or not) to the Dacey 1992
      data, which gives the receptive field size of retinal ganglion cells. This
      also uses `numpyro` and so looks fairly similar to `mcmc.py`.
    - `create_mad_images.py`: synthesize Maximally-Differentiating images (as in
      Wang and Simoncelli, 2008), to highlight mean-squared error remaining in
      human metamers.
    - `create_other_synth.py`: other ways to synthesize images to highlight
      mean-squared error remaining in human metamers.
    - `observer_model.py`: first steps towards an observer model to predict
      human performance when images are *not* metamers. Did not end up making
      much progress, so this is not present in the paper.
    - `utils.py`: various utility functions.
    - `style.py`: code for styling the figures.
  - `extra_packages/`: additional python code used by this repo. The bits that
    live here were originally part of
    [plenoptic](https://github.com/LabForComputationalVision/plenoptic/), but
    were pulled out because it's a bad idea for a research project to be so
    heavily reliant on a project currently under development.
    - `pooling-windows`: git submodule that points to [this
      repo](https://github.com/LabForComputationalVision/pooling-windows),
      containing the pooling windows we use.
    - `plenoptic_part`: contains the models and metamer synthesis code (as well
      as some utilities) that were pulled out of plenoptic, branching at [this
      commit](https://github.com/LabForComputationalVision/plenoptic/tree/fb1c4d29c645c9a054baa021c7ffd07609b181d4)
      (I used [git filter-repo](https://github.com/newren/git-filter-repo/) and
      so the history should be preserved). While the model code (and some of the
      utilities) have been deleted from `plenoptic` and are unique to this repo,
      the synthesis code here is a modified version of the one in plenoptic. If
      you wish to use synthesis for your own work *use the plenoptic version*,
      which is regularly tested and supported.
  - `notebooks/`: jupyter notebooks for investigating this project in more
    detail.
    - `Freeman_Check.ipynb`: notebook checking that our windows are the same
      size as those from Freeman and Simoncelli, 2011 (and thus that the models'
      scaling parameter has the same meaning); see
      [below](#check-against-freeman-and-simoncelli-2011-windows) for more
      details.
  - `examples_images.py`: script to open up some example images to show
    participants before the experiment (example usage elsewhere in this readme).
  - `download_data.py`: script to download and arrange data for reproducing
    results and figures. See explanation elsewhere in this readme.
  - `matlab/`: two matlab scripts using external matlab libraries. Neither are
    necessary: one is used to generate the windows from the Freeman and
    Simoncelli, 2011 paper (the output of which can be downloaded using
    `download_data.py`) and the other generates some LGN-like image statistics
    that we didn't end up using.
  - `data/`: contains some data files.
    - `Dacey1992_RGC.csv`: csv containing data from figure 2B of Dennis M. Dacey
      and Michael R. Petersen (1992), "Dendritic field size and morphology of
      midget and parasol ganglion cells of the human retina", PNAS 89,
      9666-9670, extracted using
      [WebPlotDigitizer](https://apps.automeris.io/wpd/) on July 15, 2021. To
      recreate that figure, using the snakemake rule `dacey_figure`. Note that
      we did not separate the data into nasal field and temporal, upper, and
      lower fields, as the paper does.
    - `checksums.json`: json file containing BLAKE2b hashes for the files
      downloadable via `download_data.py`, so we can check they downloaded
      corectly.
  - `reports/`: contains a variety of figure-related files.
     - `figures/`: these are figure components that I use when putting the
       figures together. They fall into two categories: schematics that are
       copied as is, with no changes (e.g., image space schematics, experiment
       schematic), and templates that we embed images into (e.g., the example
       metamer figures).
     - `paper_figures/`: these are the actual figures used in the paper, as
       created by the `snakemake` file. There are none in the github repo, see
       main repo README for details on how to create them.
     - `figure_rules.txt`: this is a list of snakemake rules that create figures
       (rather than analyze the data). It can be used to limit snakemake's
       search of possible analysis paths. See main github README for more
       details.
  - `tests/test_models.py`: contains a small number of tests of the pooling
    models, ran weekly and on every push (alongside other tests).
  - `environment-psychopy.yml`, `environment.yml`: yml files defining conda
    environment for the experiment (using `psychopy`) and for everything. See
    elsewhere in this readme for details.
  - `greene.json`, `rusty.json`: json files defining how snakemake should
    communicate with NYU's and Flatiron's SLURM clusters, respectively (works
    with the [snakemake-slurm](https://github.com/billbrod/snakemake-slurm)
    profile). See elsewhere in this readme for details.
  - `config.yml`: yml configuration file, defining paths, metmaer structure, and
    some info on experiment structure.

# Usage details

The general structure of the research project this repo describes is
as follows:

1. Develop models of the early visual system
2. Generate metamers for these models
3. Use psychophysics to set model parameters

The code for the models and general metamer synthesis are contained in the
[plenoptic library](https://github.com/LabForComputationalVision/plenoptic/);
this repo has four main components: generate metamers (2), prepare for the
experiment (3), run the experiment (3), and analyze the data from the experiment
(3). How to use this repo for each of those tasks is described below.

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
wget -O- https://osf.io/td3ea/download | tar xvz -C .
```

You should now have three directories here: `raw_images`, `ref_images` and
`norm_stats`. `raw_images` should contain four `.NEF` (Nikon's raw format)
images: `azulejos`, `flower`, `tiles`, and `market`. `norm_stats` should contain
a single `.pt` (pytorch) file: `V1_texture_degamma_norm_stats.pt`. `ref_images`
should contain `einstein_size-256,256.png`, which we'll use for testing the
setup, as well as `.tiff` versions of the four raw images (the raw images are
provided in case you want to try a different demosaicing algorithm than the one
I did; if you're fine with that step, you can ignore them and everything further
will use the `.tiff` files found in `ref_images`).

## Test setup

A quick snakemake rule is provided to test whether your setup is
working: `snakemake -j 4 -prk test_setup_all`. This will create a small number
of metamers, without running the optimization to completion. If this
runs without throwing any exceptions, your environment should be set
up correctly and you should have gpus available.

The output will end up in `~/Desktop/metamers/test_setup` and you can
delete this folder after you've finished.

## Check against Freeman and Simoncelli, 2011 windows

This project uses a modification of the pooling windows first described in
Freeman and Simoncelli, 2011. We include some code to check our reimplementation
of the windows and the extension to use Gaussians instead of raised-cosine
falloffs. Basically, we want to make sure that our windows are the same size --
identical reimplementation is not important, but we want to make sure that the
models' scaling parameter has the same interpretation; it should be the ratio
between the eccentricity and the radial diameter of the windows at half-max
amplitude. To do so, we include a notebook `notebooks/Freeman_Check.ipynb`, as
well as some snakemake rules.

We check two things: that our windows' scaling parameter has the same meaning as
that in the original paper, and that our V1 metamers look approximately the
same. You can view this by looking at the `Freeman_Check` notebook and its
cached outputs directly. If you wish to run the notebook or investigate the
objects in more detail, you can run either the `freeman_check` or
`download_freeman_check` snakemake rules (`freeman_check` will run the analyses
and so requires matlab and a GPU, while `download_freeman_check` will just
download the outputs of this from the [OSF](https://osf.io/67tbe/)):

``` sh
conda activate metamers
snakemake -prk download_freeman_check
# OR
snakemake -prk freeman_check
```

Once you've done that, you can start up the jupyter notebook. There are two main
ways of getting jupyter working so you can view the included notebook:

1. Install jupyter in this `metamers` environment: 

``` sh
conda activate metamers
conda install -c conda-forge jupyterlab
```

   This is easy but, if you have multiple conda environments and want to use
   Jupyter notebooks in each of them, it will take up a lot of space.
   
2. Use [nb_conda_kernels](https://github.com/Anaconda-Platform/nb_conda_kernels):

``` sh
# activate your 'base' environment, the default one created by miniconda
conda activate 
# install jupyter lab and nb_conda_kernels in your base environment
conda install -c conda-forge jupyterlab
conda install nb_conda_kernels
# install ipykernel in the calibration environment
conda install -n metamers ipykernel
```

   This is a bit more complicated, but means you only have one installation of
   jupyter lab on your machine.
   
In either case, to open the notebooks, navigate to the `notebooks/` directory on
your terminal and activate the environment you install jupyter into (`metamers`
for 1, `base` for 2), then run `jupyter` and open up the notebook. If you
followed the second method, you should be prompted to select your kernel the
first time you open a notebook: select the one named "metamers".

A portion of the results presented in this notebook are also found in one of the
paper's appendices.

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
up to about 8 hours. For the V1 images, the max is about three
hours. -- TODO: UPDATE THESE ESTIMATES

The more GPUs you have available, the better.

If you wanted to generate all of your metamers at once, this is very
easy: simply running

```
python -m foveated_metamers.utils RGC V1 -g | xargs snakemake -j n --resources gpu=n mem=m -prk --restart-times 3 --ri
```

will do this (where you should replace both `n` with the number of
GPUs you have; this is how many jobs we run simultaneously; assuming
everything is working correctly, you could increase the `n` after `-j`
to be greater than the one after `--resources gpu=`, and snakemake
should be able to figure everything out; you should also replace `m`
with the GB of RAM you have available). `snakemake` will create the
directed acyclic graph (DAG) of jobs necessary to create all metamers.

However, you probably can't create all metamers at once on one machine, because
that would take too much time. You probably want to split things up. If you've
got a cluster system, you can configure `snakemake` to work with it in a
[straightforward
manner](https://snakemake.readthedocs.io/en/stable/executable.html#cluster-execution)
(snakemake also works with cloud services like AWS, kubernetes, but I have no
experience with that; you should google around to find info for your specific
job scheduler, see the small repo [I put
together](https://github.com/billbrod/snakemake-slurm) for using NYU's or the
Flatiron Institute's SLURM system). In that case, you'll need to put together a
`cluster.json` file within this directory to tell snakemake how to request GPUs,
etc (see `greene.json` and `rusty.json` for the config files I use on NYU's and
Flatiron's, respectively). Something like this should work for a SLURM system
(the different `key: value` pairs would probably need to be changed on different
systems, depending on how you request resources; the one that's probably the
most variable is the final line, gpus):

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
make sure you don't run five jobs that require 1 gpus each if you only
have 4 gpus total, for example. Similarly, `resources.mem` provides an
estimate of how much memory (in GB) the job will use, which we use
similarly when requesting resources above. This is just an estimate
and, if you find yourself running out of RAM, you may need to increase
it in the `get_mem_estimate` function in `Snakefile.`

If you don't have a cluster available and instead have several machines with
GPUs so you can split up the jobs, making use of the
`foveated_metamers/utils.py` script. See it's help string for details, but
calling it from the command line with different arguments will generate the
paths for the corresponding metamers. For example, to print the path to all RGC
metamers with a given scaling value, you would run

```
python -m foveated_metamers.utils RGC -g --scaling 0.01
```

The `-g` argument tells the script to include the gamma-correction step (for
viewing on non-linear displays) and `RGC` and `--scaling 0.01` tell it to
use that model and scaling value, respectively. While messing with this, pay
attention to the `-j n` and `--resources gpu=n` flags, which tell snakemake how
many jobs to run at once and how many GPUs you have available, respectively.

Note that I'm using dotlockfile to handle scheduling jobs across
different GPUs. I think this will work, but I recommend adding the
`--restart-times 3` flag to the snakemake call, as I do above, which
tells snakemake to try re-submitting a job up to 3 times if it
fails. Hopefully, the second time a job is submitted, it won't have a
similar problem. But it might require running the `snakemake` command
a small number of times in order to get everything straightened out.

## Prepare for experiment

Once the metamers have all been generated, they'll need to be combined
into a numpy array for the displaying during the experiment, and the
presentation indices will need to generated for each subject.

For the experiment we performed, we had 8 subjects, with 3 sessions per model
per comparison, each containing 5 original images and lasting an hour. In order
to re-generate the indices we used, you can simply run `snakemake -prk
gen_all_idx`. This will generate the indices for each subject, each session,
each model, as well as the stimuli array (we actually use the same index for
each model for a given subject and session number; they're generated using the
same seed).

This can be run on your local machine, as it won't take too much time
or memory.

The stimuli arrays will be located at:
`~/Desktop/metamers/stimuli/{model_name}/stimuli_comp-{comp}.npy` and the presentation
indices will be at
`~/Desktop/metamers/stimuli/{model_name}/task-split_comp-{comp}/{subj_name}/{subj_name}_task-split_comp-{comp}_idx_sess-{sess_num}_im-{im_num}.npy`,
where `{comp}` is `met` and `ref`, for the metamer vs metamer and metamer vs
reference image comparisons, respectively. There will also be a pandas
DataFrame, saved as a csv, at
`~/Desktop/metamers/stimuli/{model_name}/stimuli_description.csv`, which
contains information about the metamers and their optimization. It's used to
generate the presentation indices as well as to analyze the data.

You can generate your own, novel presentation indices by running `snakemake -prk
~/Desktop/metamers/stimuli/{model_name}/task-split_comp-{comp}/{subj_name}/{subj_name}_task-split_comp-{comp}_idx_sess-{sess_num}_run-{run_num}.npy`,
replacing `{model_name}` with one of `'RGC_norm_gaussian',
'V1_norm_s6_gaussian'`, `{subj_name}` must be of the format `sub-##`, where `##`
is some integer (ideally zero-padded, but this isn't required), `{sess_num}`
must also be an integer (this is because we use the number in the subject name
and session number to determine the seed for randomizing the presentation order;
if you'd like to change this, see the snakemake rule `generate_experiment_idx`,
and how the parameter `seed` is determined; as long as you modify this so that
each subject/session combination gets a unique seed, everything should be fine),
`{run_num}` is `00` or `01` (determines which set of 4 reference images are
shown), and `comp` take one of the values explained above.

### Demo / test experiment

For teaching the subjects about the task, we have two brief training runs: one
with noise images and one with a small number of metamers. To put them together,
run `snakemake -prk
~/Desktop/metamers/stimuli/training_noise/task-split_comp-met/sub-training/sub-training_task-split_comp-met_idx_sess-00_run-00.npy
~/Desktop/metamers/stimuli/training_RGC_norm_gaussian/task-split_comp-met/sub-training/sub-training_task-split_comp-met_idx_sess-00_run-00.npy
~/Desktop/metamers/stimuli/training_V1_norm_s6_gaussian/task-split_comp-met/sub-training/sub-training_task-split_comp-met_idx_sess-00_run-00.npy`.
This will make sure the stimuli and index files are created. Then run the
[training](#training) section below.

## Run experiment

To run the experiment, make sure that the stimuli array and presentation indices
have been generated and are at the appropriate path. It's recommended that you
use a chin-rest or bite bar to guarantee that your subject remains fixated on
the center of the image; the results of the experiment rely very heavily on the
subject's and model's foveations being identical.

We want 6 sessions per subject per model. Each session will contain all trials
in the experiment for 4 of the 8 reference images, the only thing that differs
is the presentation order (each subject gets presented a different split of 4
reference images). 

### Training

To teach the subject about the experiment, we want to introduce them to the
structure of the task and the images used. The first one probably only needs to
be done the first time a given subject is collecting data for each model, the
second should be done at the beginning of each session.

1. First, run a simple training run (make sure the stimuli and indices are
   created, as described [above](#demo--test-experiment)):
    - `conda activate psypy` 
    - `python foveated_metamers/experiment.py ~/Desktop/metamers/stimuli/training_noise/stimuli_comp-{comp}.npy sub-training 0 -s 0 -c {comp} ; python foveated_metamers/experiment.py ~/Desktop/metamers/stimuli/training_{model}/stimuli_comp-{comp}.npy sub-training 0 -s 0 -c {comp}` 
       where `{comp}` is `met` or `ref`, depending on which version you're
       running, and `{model}` is `RGC_norm_gaussian` or `V1_norm_s6_gaussian`,
       depending on which you're running.
    - Explanatory text will appear on screen, answer any questions.
    - This will run two separate training runs, both about one or two minutes,
      each followed by feedback. 
    - The first one will just be comparing natural to noise images and so the
      subject should get 100%. The goal of this one is to explain the basic
      structure of the experiment.
    - The second will have two metamers, one easy and one hard, for each of two
      reference images. They should get 100% on the easy one, and do worse on
      the hard. The goal of this one is to show what the task is like with
      metamers and give them a feeling for what they may look like.
2. Run: 
   - `conda activate metamers`
   - `python example_images.py {model} {subj_name} {sess_num}` where `{model}`
     is `V1` or `RGC` depending on which model you're running, and `{subj_name}`
     and `{sess_num}` give the name of the subject and number of this session,
     respectively.
   - This will open up three image viewers. Each has all 5 reference images the
     subject will see this session. One shows the reference images themselves,
     one the metamers with the lowest scaling value, and one the metamers with
     the highest scaling value (all linear, not gamma-corrected).
    - Allow the participant to flip between these images at their leisure, so
      they understand what the images will look like.

### Split-screen Task

Now, using a split-screen task. Each trial lasts 1.4 seconds and is structured
like so:

```
|Image 1 | Blank  |Image 2 |Response|  Blank |
|--------|--------|--------|--------|--------|
|200 msec|500 msec|200 msec|        |500 msec|
```

Image 1 will consist of a single image divided vertically at the center by a
gray bar. One half of image 2 will be the same as image 1, and the other half
will have changed. The two images involved are either two metamers with the same
scaling value (if `comp=met`) or a metamer and the reference image it is based
on (if `comp=ref`). The subject's task is to say whether the left or the right
half changed. They have as long as they need to respond and receive no feedback.

To run the experiment:

- Activate the `psypy` environment: `conda activate psypy`
- Start the experiment script from the command line: 
   - `python foveated_metamers/experiment.py ~/Desktop/metamers/stimuli/{model}/stimuli_comp-{comp}.npy {subj_name} {sess_num} -c {comp}` 
     where `{model}, {subj_name}, {sess_num}, {comp}` are as described in the
     [training](#training) section.
   - There are several other arguments the experiment script can take,
     run `python foveated_metamers/experiment.py -h` to see them, and
     see the [other arguments](#other-arguments) section for more
     information.
- Explain the task to the subject, as seen in the "say this to subject for
  experiment" section (similar text will also appear on screen before each run
  for the participant to read)
- When the subject is ready, press the space bar to begin the task.
- You can press the space bar at any point to pause it, but the pause
  won't happen until the end of the current trial, so don't press it a
  bunch of times because it doesn't seem to be working. However, try
  not to pause the experiment at all.
- You can press q/esc to quit, but don't do this unless truly
  necessary.
- There will be a break half-way through the block. The subject can
  get up, walk, and stretch during this period, but remind them to
  take no more than a minute. When they're ready to begin again,
  press the space bar to resume.
- The data will be saved on every trial, so if you do need to quit out, all is
  not lost. If you restart from the same run, we'll pick up where we left off.
- The above command will loop through all five runs for a given session. To do a
  particular set of runs pass `-r {run_1} {run_2} ... {run_n}` to the call to
  `experiment.py` (where `{run_1}` through `{run_n}` are 0-indexed integers
  specifying the runs to include). For example, if you quit out on the third run
  and wanted to finish that one and then do runs 4 and 5, pass: `-r 2 3 4`. If
  you just want to finish that run, you'd only pass `-r 2`.

Recommended explanation to subjects:

> In this experiment, you'll be asked to complete what we call an "2-Alternative
> Forced Choice task": you'll view an image, split in half, and then, after a
> brief delay, a second image, also split in half. One half of the second image
> will be the same as the first, but the other half will have changed. Your task
> is to press the left or right button to say which half you think changed. All
> the images will be presented for a very brief period of time, so pay
> attention. Sometimes, the two images will be very similar; sometimes, they'll
> be very different. For the very similar images, we expect the task to be hard.
> Just do your best!

> You'll be comparing natural and synthesized images. The first image can be
> either natural or synthesized, so pay attention! You will receive no feedback,
> either during or after the run.

> For this experiment, fixate on the center of the image the whole time and try
> not to move your eyes.

> The run will last for about twelve minutes, but there will be a break halfway
> through. During the break, you can move away from the device, walk around, and
> stretch, but please don't take more than a minute. 

This part will not be shown on screen, and so is important:

> You'll complete 5 runs total. After each run, there will be a brief pause, and
> then the instruction text will appear again, to start the next run. You can
> take a break at this point, and press the spacebar when you're ready to begin
> the next run.

### Other arguments

The `experiment.py` takes several optional arguments, several of which
are probably relevant in order to re-run this on a different
experiment set up:

- `--screen` / `-s`: one integer which indicate which screens
  to use. 
- `--screen_size_pix` / `-p`: two integers which indicate the size of
  the screen(s) in pixels .
- `--screen_size_deg` / `-d`: a single float which gives the length of
  the longest screen side in degrees.

For more details on the other arguments, run `python
foveated_metamers/experiment.py -h` to see the full docstring.

NOTE: While the above options allow you to run the experiment on a setup that
has a different screen size (both in pixels and in degrees) than the intended
one, the metamers were created with this specific set up in mind. Things should
be approximately correct on a different setup (in particular, double-check that
images are cropped, not stretched, when presented on a smaller monitor), but
there's no guarantee. If you run this experiment, with these stimuli, on a
different setup, my guess is that the psychophysical curves will look different,
but that their critical scaling values should approximately match; that is,
there's no guarantee that all scaling values will give images that will be
equally confusable on different setups, but the maximum scaling value that leads
to 50% accuracy should be about the same. The more different the viewing
conditions, the less likely that this will hold.

### Checklist

The following is a checklist for how to run the experiment. Print it out and
keep it by the computer.

Every time:

1. Make sure monitor is using the correct icc profile (`linear-profile`;
   everything should look weirdly washed out). If not, hit the super key (the
   Windows key on a Windows keyboard) and type `icc`, open up the color manager
   and enable the linear profile.
   
First session only (on later sessions, ask if they need a refresher):
   
2. Show the participant the set up and show the participants the wipes and say
   they can use them to wipe down the chinrest and button box.
   
3. Tell the participant:

> In this task, a natural image will briefly flash on screen, followed by a gray
> screen, followed by another image. Half of that second image will be the same
> as the first, half will have changed. Your task is to say which half has
> changed, using these buttons to say "left" or "right". You have as long as
> you'd like to respond, and you will not receive feedback. There will be a
> pause halfway through, as well as between runs; take a break and press the
> center button (labeled "space") to continue when you're ready. You won't press
> the buttons in the bottom row.

4. Train the participant. Say:

> Now, we'll do two brief training runs, each of which will last about a minute.
> In the first, you'll be comparing natural images and noise; the goal is so you
> understand the basic structure of the experiment. In the second, you'll be
> comparing those same natural images to some of the stimuli from the
> experiment; some will be easy, some hard. You'll receive feedback at the end
> of the run, to make sure you understand the task. I'll remain in the room to
> answer any questions.
>
> There will be fixation dot in the center of some explanatory text at the
> beginning, use that to center yourself.

5. Run (replace `{model}` with `V1_norm_s6_gaussian` or `RGC_norm_gaussian`):

``` sh
conda activate psypy
python foveated_metamers/experiment.py ~/Desktop/metamers/stimuli/training_noise/stimuli_comp-ref.npy sub-training 0 -c ref ; python foveated_metamers/experiment.py ~/Desktop/metamers/stimuli/training_{model}/stimuli_comp-ref.npy sub-training 0 -c ref
```

6. Answer any questions.

Every time:

7. Show the participant the images they'll see this session, replacing `{model}`
   with `V1` or `RGC` (no need to use the full name), and `{subj_name}` and
   `{sess_num}` as appropriate:

``` sh
conda activate metamers
python example_images.py {model} {subj_name} {sess_num}
```

8. Say the following and answer any questions:

> These are the natural images you'll be seeing this session, as well as some
> easy and hard stimuli. You can look through them for as long as you'd like.
 
9. Ask if they have any questions before the experiment.

10. Say:

> This will run through all 5 runs for this session. Each should take you 10 to
> 12 minutes. Come get me when you're done. As a reminder, you have as long as
> you'd like to respond, and you won't receive any feedback.

10. Run, replacing `{model}`, `{subj_name}`, `{sess_num}` as above:

``` sh
conda activate psypy
python foveated_metamers/experiment.py ~/Desktop/metamers/stimuli/{model}/stimuli_comp-ref.npy {subj_name} {sess_num} -c ref
```

## Analyze experiment output

*In progress*

# Known issues

1. When using multiprocessing (as done when fitting the psychophysical curves)
   from the command-line, I get `OMP: Error #13: Assertion faliure at
   z_Linux_util.cpp(2361)` on my Ubuntu 18.04 laptop. As reported
   [here](https://github.com/ContinuumIO/anaconda-issues/issues/11294), this is
   a known issue, and the solution appears to be to set an environmental
   variable: running `export KMP_INIT_AT_FORK=FALSE` in the open terminal will
   fix the problem. Strangely, this doesn't appear to be a problem in a Jupyter
   notebook, but it does from `IPython` or the `snakemake` calls. I tried to set
   the environmental variable from within Snakefile, but I can't make that work.
   Running the calls with `use_multiproc=False` will also work, though it will
   obviously be much slower.
2. When trying to use the `embed_bitmaps_into_figure` rule on a drive mounted
   using `rclone` (I had my data stored on a Google Drive that I was using
   `rclone` to mount on my laptop), I got a `'Bad file descriptor'` error from
   python when it tried to write the snakemake log at the end of the step. It
   appears to be [this
   issue](https://forum.rclone.org/t/bad-file-descriptor-when-moving-files-to-rclone-mount-point/13936),
   adding the `--vfs-cache-mode writes` flag to the `rclone mount` command
   worked (though I also had to give myself full permissions on the rclone cache
   folder: `sudo chmod -R 777 ~/.cache/rclone`).
   
# Notes on reproducibility

The intention of sharing this code is to allow for the reproduction of the
figures in the resulting paper. This is the code I used to synthesize the
metamers used in the experiment, and you can use it to do so, but there are a
couple things you should be aware of:

- Results will not be identical on CPUs and GPUs. See PyTorch's
  [notes](https://pytorch.org/docs/stable/notes/randomness.html) on this.
- I used stochastic weight averaging (SWA) for the energy model metamers. SWA
  seems to reduce the final loss by averaging pixel values as we get near
  convergence (see
  [here](https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/)
  for more details). However, the version of SWA I used to generated the
  metamers for the experiment was from `torchcontrib`, which was archived in
  2020 and is no longer maintained ([github
  repo](https://github.com/pytorch/contrib)). In May 2022, I noticed that the
  torchcontrib SWA implementation no longer worked on my tests with the most
  recent versions of python (3.10) and pytorch (1.12), so I [updated my
  code](https://github.com/billbrod/foveated-metamers/pull/3) to work with the
  pytorch SWA implementation. The resulting metamers are not identical to the
  ones produced before, but they are similar in both visual quality and loss,
  and I believe they would be indistinguishable in a 2AFC task.
- The metamer synthesis code found here (in `extra_packages/plenoptic_part`) was
  very much a work in progress throughout this whole project and ended up
  becoming a tangled rats nest, as is the case for most research code.
  
For all the above reasons, I am sharing the synthesized metamers used in this
experiment and recommend you use them directly if you need the exact images I
used (to replicate my behavioral results, for example). If you wish to
synthesize new metamers, whether using your own model or even using the ones
from this paper, I strongly recommend you use the metamer synthesis code found
in [plenoptic](https://github.com/LabForComputationalVision/plenoptic/), which
is actively maintained and tested, though it is not identical to the procedure
used here. Most important, it does not include a SWA implementation and probably
will never include one, but I would be happy to help come up with how to add it
in an extension or a fork.
   
# Related repos

If you would like to generate your own metamers, see
[plenoptic](https://github.com/LabForComputationalVision/plenoptic/), a python
library for image synthesis, including metamers, MAD Competition,
eigendistortions, and geodesics.

If you would like to use the pooling windows, see
[pooling-windows](https://github.com/LabForComputationalVision/pooling-windows).
This includes pytorch implementations of the Gaussian windows from this project,
as well as the raised-cosine windows from Freeman and Simoncelli, 2011. The
README describes how to use them for creating a version of the pooled luminance
and energy models used in this project. Feel free to use the versions in this
repo, but the simpler version from that README may better suit your needs. The
version in this repo includes a bunch of helper code, including for creating
plots and the starts of paths not taken. The only important thing missing from
the `pooling-windows` repo is normalization -- look for the `normalize_dict`
attribute in `extra_packages/plenoptic_part/simulate/ventral_stream.py` to see
how I implemented that.

jeremy's code, Wallis code
   
# Citation

If you use the data or code (including the stimuli) from this project in an
academic publication, please cite the [poster](https://osf.io/aketq/).

More to come...

# References

- Freeman, J., & Simoncelli, E. P. (2011). Metamers of the ventral
  stream. Nature Neuroscience, 14(9),
  1195–1201. http://dx.doi.org/10.1038/nn.2889

- Wang, Z., & Simoncelli, E. P. (2008). Maximum differentiation (MAD)
  competition: A methodology for comparing computational models of perceptual
  discriminability. Journal of Vision, 8(12), 1–13.
  http://dx.doi.org/10.1167/8.12.8
