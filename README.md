# foveated-metamers

[![DOI](https://zenodo.org/badge/195451020.svg)](https://zenodo.org/badge/latestdoi/195451020)

This repo contains the code for a vision science experiment investigating how
human perception changes across the visual field using behavioral experiments
and computational models inspired by the earlys tages of visual processing. We
use these models to investigate what people cannot see, an approach that has a
long history of vision science. If we know what information people are
insensitive to, we can discard it or randomize it, and the resulting image
should appear unchanged from the original.

See the [preprint](https://doi.org/10.1101/2023.05.18.541306) or the [VSS 2023
poster](https://osf.io/8hdaz/) for scientific details. You may also be
interested in the
[website](https://users.flatironinstitute.org/~wbroderick/metamers/) we put
together for browsing through the synthesized images, and the
[OSF](https://osf.io/67tbe/) for bulk downloading the images or the behavioral
data. Finally, you may be interested in
[plenoptic](https://plenoptic.readthedocs.io/en/latest/), a software package for
generating metamers (and more!) for your own models.

If you re-use some component of this project in an academic publication, see the
[citation](#citation) section for how to credit us.

Table of Contents
=================

* [Usage](#usage)
   * [What if I want to do more than recreate the figures?](#what-if-i-want-to-do-more-than-recreate-the-figures)
      * [... examine the metamers synthesized for this project](#-examine-the-metamers-synthesized-for-this-project)
      * [... synthesize some metamers](#-synthesize-some-metamers)
      * [... see what the experiment was like](#-see-what-the-experiment-was-like)
      * [... run the full experiment](#-run-the-full-experiment)
      * [... refit the psychophysical curves](#-refit-the-psychophysical-curves)
* [Setup](#setup)
   * [Software requirements](#software-requirements)
      * [Python](#python)
      * [Jupyter](#jupyter)
      * [Experiment environment](#experiment-environment)
   * [Data](#data)
      * [Source images](#source-images)
      * [Download data](#download-data)
   * [config.yml](#configyml)
* [Directory structure](#directory-structure)
* [Usage details](#usage-details)
   * [Check against Freeman and Simoncelli, 2011 windows](#check-against-freeman-and-simoncelli-2011-windows)
   * [Experiment notes](#experiment-notes)
      * [Training](#training)
      * [Task structure](#task-structure)
      * [Other arguments](#other-arguments)
      * [Experiment Checklist](#experiment-checklist)
* [Known issues](#known-issues)
* [Notes on reproducibility](#notes-on-reproducibility)
* [Related repos](#related-repos)
* [Citation](#citation)
* [References](#references)

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
    - Because we use git submodules, you'll also need to run the following two
      lines:
      ```sh
      git submodule sync
      git submodule update --init --recursive
      ```
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
   - Additionally, install [inkscape](https://inkscape.org/), version equal to
     or greater 1.0.2. It seems like that, with inkscape version 1.2.2, all
     images are embedded at full resolution, leading to a massive increase in
     file size. I'm not sure what change causes this. It works as intended with
     inkscape 1.1
   - Check if you have `rsync` available (you probably do) and install it if you
     don't (probably best to do so via a package manager).
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
2. The complete workflow is very long (going back to preparing the images for
   metamer synthesis), and so sometimes `snakemake` can take a long time to
   determine what to run. This problem can get exacerbated if the file
   modification timestamps get thrown off, so that `snakemake` thinks it needs
   to re-create some of the existing files. To limit the search space and force
   `snakemake` to only consider figure-related rules, use the included
   `reports/figure_rules.txt` and the `--allowed-rules` flag: `cat
   reports/figure_rules.txt | xargs snakemake -prk main_paper_figures
   --allowed-rules`. You can pass any argument to `snakemake`, as long as the
   command ends with `--allowed-rules`.
3. Several of the figures in the paper (e.g., figure 4) include example metamers
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
      squares where the images should go. When embedding the images, we correct
      the paths, so this is not a problem.
    - It is possible that `snakemake` will get confused when you switch machines
      and decide that it wants to re-run steps because the file modification
      timestamps appear out of order (this might happen, in particular, because
      of `TEXTURE_DIR`, which is used at the very beginning of the workflow;
      point it to something old or non-existant to avoid this!). To prevent
      this, use the same trick as above: append `--allowed-rules
      embed_bitmaps_into_figures main_figures appendix_figures` to any
      `snakemake` command to ensure that it will only run the embedding rule
      (and the `main_figures` / `appendix_figures` rules).
   
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

You can also use the `foveated_metamers.utils.generate_metamer_paths` function
to generate the list of paths that I used for metamers in this experiment (it
can also be called from the command-line: `python -m foveated_metamers.utils`).
See the command-line's helpstring or the function's docstring for details, but
here's an example: in order to get the path for one of the energy model metamers
created with the llama target image for each scaling value used in the original
vs. synth white noise comparison, run: `python -m foveated_metamers.utils V1
--ref_image llama --seed_n 0`. Note that this prints out the files on a single
line, so you may want to redirect the output to a file for later viewing (append
`> mets.txt`) or split it up for easier viewing (append `| tr ' ' '\n'`).

To recreate any of my metamers, you'll also need to download the
`synthesis_input` data set, which includes the target images we used, as well as
statistics used for normalizing the models' representation.

You should also be aware that the pooling windows are very large once you get
below `scaling=0.1`, so I would start with a larger window size. It is also
strongly recommended to use a GPU, which will greatly speed up synthesis.

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

Note that the stimuli won't be rescaled by default: unless you have the exact
same experimental set up as me (screen size 3840 by 2160 pixels, viewing
distance of 40cm, 48.5 pixels per degree), you'll need to use the `-p` and
`-d` flags when calling `experiment.py` to specify the size of your screen in
pixels and degrees.

Also note that your monitor should be gamma-corrected to have a linear
relationship between luminance and pixel value.

### ... run the full experiment

First, Set up your environment for the [experiment](#experiment-environment) and
download the stimuli: `python download_data.py stimuli`. 

You may also want to download the files used in the training and [try that
out](#see-what-the-experiment-was-like).

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

Note that the stimuli won't be rescaled by default: unless you have the exact
same experimental set up as me (screen size 3840 by 2160 pixels, viewing
distance of 40cm, 48.5 pixels per degree), you'll need to use the `-p` and
`-d` flags when calling `experiment.py` to specify the size of your screen in
pixels and degrees.

And note that the above options allow you to run the experiment on a setup that
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

Also note that your monitor should be gamma-corrected to have a linear
relationship between luminance and pixel value.

### ... refit the psychophysical curves

Follow the first three steps in the [usage](#usage) section, so that you have
setup your environment and specified `DATA_DIR`. Then, download the behavioral
data: `python download_data.py behavioral_data`.

From here, it's up to you. If you'd like to use your own procedure, the [OSF
readme](https://osf.io/kjf75) describes the most relevant columns of the
behavioral `.csv` files.

If you'd like to use my procedure (using MCMC to fit a hierarchical Bayesian
model, fitting all images and subjects simultaneously but each metamer model and
comparison separately), you can do so using `snakemake`: `snakemake -prk
{DATA_DIR}/mcmc/{model_name}/task-split_comp-{comp}/task-split_comp-{comp}_mcmc_{mcmc_model}_step-{step_size}_prob-{accept_prob}_depth-{tree_depth}_c-{num_chains}_d-{num_draws}_w-{num_warmup}_s-{seed}.nc`,
where:

- `{DATA_DIR}`: the `DATA_DIR` field from the `config.yml` file 
- `{model_name}`: either `RGC_norm_gaussian` (for the luminance model) or
  `V1_norm_s6_gaussian` (energy)
- `{comp}`: one of `met`, `ref`, `met-natural`, `ref-natural` or
  `ref-downsample-2`, depending on which comparison you wish to fit.
- `{mcmc_model}`: which hierarchical model to fit to the data.
  `partially-pooled` is used in the main body of the paper, see the Methods
  section for details on it and appendix 6 for details on the other two. Only
  the `met` and `ref` comparisons have more than one subject, so the models are
  interchangeable for the rest.
  - `partially-pooled`: fit image-level and subject-level effects, no
    interaction between them.
  - `unpooled`: fit each psychophysical curve separately, no group effects.
  - `partially-pooled-interactions`: fit image-level and subject-level effects,
    plus an interation term.
- `{seed}`: computational seed for reproducibility, must be an integer.
- the rest are all hyperparameters for MCMC, see numpyro
  [documentation](https://num.pyro.ai/en/latest/mcmc.html#nuts) and
  [examples](https://num.pyro.ai/en/latest/examples/baseball.html) for more
  details.

Note that I've had issues guaranteeing exact reproducibility, even with the same
seed. I have not been able to track down why this is.

If you refit the curves using MCMC, you'll want to check the diagnostics,
effective sample size (ESS) and $\hat{R}$. You can create a plot summarizing
these for each variable with snakemake (just replace `.nc` with
`_diagnostics.png`). [Vethari et al., 2021](https://doi.org/10.1214/20-BA1221)
recommend that you look for $\hat{R} < 1.01$ for all variables.

You can download my fits to the data (`python download_data.py mcmc_fits` for
the partially pooled model, `python download_data.py mcmc_compare` for the other
two) or use `foveated_metamers.utils.get_mcmc_hyperparams` to see what
hyper-parameters I used to fit each comparison.

# Setup

The analyses were all run on Linux (Ubuntu, Fedora, and CentOS, several
different releases). Everything should work on Macs. For Windows, I would
suggest looking into the [Windows Subsystem for
Linux](https://docs.microsoft.com/en-us/windows/wsl/about), as Windows is very
different from the others.

## Software requirements

The [python environment](#python) discussed below are required regardless of
what you wish to do. There are several additional requirements that are
optional:
  - If you are using `download_data.py`, you'll need `rsync`, which you probably
    already have. If not, it's probably best to grab it via your OS's package
    manager.
  - If you are synthesizing metamers, you'll need `ffmpeg`. This is probably
    also already installed on your machine, but if not, [a static
    build](https://www.johnvansickle.com/ffmpeg/faq/) is probably the easiest
    way to go.
  - If you're using GPUs for image synthesis, you'll need the python package
    `dotlockfile` as well: `pip install dotlockfile`.
  - For embedding images into our figures (the last step of figure creation for
    about a third of the figures), you need [inkscape](https://inkscape.org),
    version equal to or greater than 1.0.2.
    - We also need to know the location of your inkscape preference file. The
      default is probably correct, but see section [config.yml](#configyml) for
      more details.
  - Appendix 3 contains a small comparison between the pooling windows used in
    this project and those found in Freeman and Simoncelli, 2011. If you wish to
    generate these windows yourself, you will need MATLAB along with [Jeremy
    Freeman's metamer code](https://github.com/freeman-lab/metamers/) and the
    [matlabPyrTools](https://github.com/LabForComputationalVision/matlabPyrTools)
    toolbox. You may also download the windows from the OSF (`python
    download_data freeman2011_check_output`). See
    [below](#check-against-freeman-and-simoncelli-2011-windows) for more
    details.

There's a [separate python environment](#experiment-environment) for running the
environment, so install that if you're planning on running the experiment.

### Python

This has been run and tested with version 3.7, unsure about more recent
versions.

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) with the
   appropriate python version.
2. Install [mamba](https://mamba.readthedocs.io/en/latest/installation.html):
   `conda install mamba -n base -c conda-forge` (I recommend using mamba instead
   of conda to install the environment because conda tends to hang while
   attempting to solve the environment).
3. After cloning this repo, run the following from this directory to grab the git submodules:
   ```sh
   git submodule sync
   git submodule update --init --recursive
   ```
4. In this directory, run `mamba env create -f environment.yml`
5. If everything works, type `conda activate metamers` to activate this
   environment.
   
As of fall 2022, the packages that I know have version requirements are
specified in `environment.yml`. However, in case updates to the dependencies
break something, I've included the output of the `pip freeze` command from two
different machines (my laptop and Flatiron's compute cluster) at
`reports/pip_freeze_laptop.txt` and `reports/pip_freeze_hpc.txt`, so a working
environment is cached (note that this includes more packages than specified in
`environment.yml`, because it includes all *their* dependencies as well).

I am also running tests on [github
actions](https://github.com/billbrod/foveated-metamers/actions) to check that
the following parts of the workflow run: the data download script, the included
notebook, metamer synthesis, the pooling models, and fitting the MCMC curves.
For all of these, I'm just testing that they *can run*, not the full workflow
completes or that I get the proper outcome, but that should be sufficient to see
what package versions are compatible. (I'm not testing figure creation, because
I couldn't come up with a way to do that didn't require more storage than is
available for free from Github runners --- if you have a solution for this, I'd
love to hear it).

### Jupyter

This repo includes [one
notebook](#check-against-freeman-and-simoncelli-2011-windows), for examining the
differences between this project and Freeman and Simoncelli, 2011. You can view the cached output of the notebook online, or you can install jupyter locally and run the notebook.

There are two main ways of installing jupyter locally:

1. Install jupyter in this `metamers` environment: 

``` sh
conda activate metamers
mamba install -c conda-forge jupyterlab
```

   This is easy but, if you have multiple conda environments and want to use
   Jupyter notebooks in each of them, it will take up a lot of space.
   
2. Use [nb_conda_kernels](https://github.com/Anaconda-Platform/nb_conda_kernels):

``` sh
# activate your 'base' environment, the default one created by miniconda
conda activate 
# install jupyter lab and nb_conda_kernels in your base environment
mamba install -c conda-forge jupyterlab
mamba install nb_conda_kernels
# install ipykernel in the calibration environment
mamba install -n metamers ipykernel
```

   This is a bit more complicated, but means you only have one installation of
   jupyter lab on your machine.

### Experiment environment

Install miniconda and mamba as described [above](#python) (you probably don't
need the submodules, but they won't hurt), then run:

```
mamba env create -f environment-psychopy.yml
```

Then, to activate, run `conda activate psypy`.

PsychoPy provides multiple backends. I'm now using the `pyglet` backend, but
I've occasionally had issues with a weird [`XF86VidModeGetGammaRamp failed`
error](https://github.com/psychopy/psychopy/issues/2061). That's just something
to be aware of.

This `environment-psychopy.yml` file pins all the package versions but I didn't
spend a lot of time trying to figure out what exactly was necessary or worry
about the versions overly-much. That is to say, I think as long as you can get a
working psychopy install with the `pyglet` backend, the code should work. So you
could try just to install `psychopy`, `pyglet`, `h5py` (used for storing
behavioral data) and `numpy` (stimuli are saved as numpy arrays) and go from
there.

## Data

### Source images

We use images from the authors' personal collection and the [UPenn Natural Image
Database](http://tofu.psych.upenn.edu/~upennidb/) as the targets for our metamer
generation. This is because we need images that are large, linear (i.e., their
pixel intensities are proportional to photon count, as you get from an image
that has not been processed in any way), and openly-licensed.

- Authors' personal collection: 
    - WFB: azulejos, tiles, bike, graffiti, llama, terraces
    - EPS: ivy, nyc, rocks, boats, gnarled, lettuce
- UPenn Natural Image Database: treetop (cd01A/DSC_0033), grooming
  (cd02A/DSC_0011), palm (cd02A/DSC_0043), leaves (cd12A/DSC_0030), portrait
  (cd58A/DSC_0001), troop (cd58A/DSC_0008).
- Unpublished photos from David Brainard: quad (EXPOSURE_ASC/DSC_0014), highway
  (SNAPSHOTS/DSC_0200).

`.tiff` files of all these images can be downloaded from the [OSF
page](https://osf.io/e2y4v/). They have been demosaiced (using DCRAW) but
otherwise untouched from the raw files. For image synthesis they were converted
to 16-bit grayscale png files, cropped / expanded to 2048 by 2600 pixels (Prof.
Brainard's photos and those from the UPenn Natural Image Database were 2014
pixels tall and so a small amount of reflection padding was used on these
photos), and rescaled so all pixel values lay between .05 and .95. These png
files are found in the `synthesis_input` tarball on the [OSF
page](https://osf.io/sw4tb).

### Download data

The data for this project is available on its [OSF page](https://osf.io/67tbe/)
(almost everything) and NYU's [Faculty Digital
Archive](https://archive.nyu.edu/handle/2451/63953) (the files required for
appendix 6, which are the fits using the alternative MCMC models). The [OSF
readme](https://osf.io/kjf75) describes its contents in detail.

We also provide the `download_data.py` script, which downloads and arranges the
data into the structure that `snakemake` expects. The following data sets can be
downloaded with this script:

1. `synthesis_input`: the inputs required for synthesizing model metamers, this
   includes the original images whose model representation our metamers match
   and the set of statistics used to normalize model responses.
2. `stimuli`: numpy arrays containing the model metamers used as stimuli in our
   experiment for each model and comparison separately, along with csv files
   with their metadata.
3. `behavioral_data`: csv files containing participant responses, along with
   metadata (each model and comparison separate).
4. `mcmc_fits`: `.nc` files containing the fits of the partially pooled
   hierarchical model to behavior, for each model and comparison separately
5. `figure_input`: miscellaneous files not contained in the other data sets
   required to create the figures in the paper (some of the model metamers,
   etc.)
6. `freeman2011_check_input`, `freeman2011_check_output`: the input and output
   files for a brief comparison between our pooling windows and energy model
   metamers and those presented in Freeman and Simoncelli, 2011. See
   [below](#check-against-freeman-and-simoncelli-2011-windows) for more details.
7. `experiment_training`: some files used to train participants, can also be
   viewed in order to get a sense for how the experiment was structured. See
   [above](#see-what-the-experiment-was-like) for details.
8. `mcmc_compare`: `.nc` files containing the fits of the alternative MCMC
   models to behavior, as well as the csv files evaluating each MCMC models
   performance, used for appendix 6.

To download one of the above, call `python download_data.py {TARGET_DATASET}`,
replacing `{TARGET_DATASET}` with one of the above. Make sure you have specified
`DATA_DIR` in `config.yml` first. The script will give you an estimate of how
large each data set is and ask if you would like to continue.

There are several components found on the OSF page that cannot be downloaded
using `download_data.py`. These are made available because they might be useful
for others, but I do not expect them to be used by anyone with this repo,
because they are not necessary for reproducing the figures.

## config.yml

This is configuration file containing options used by `snakemake`. You need to
modify the first one, `DATA_DIR`, and might need to modify the other paths in
that first section. All the other options should be left unmodified. The file is
commented explaining what each of the options are.

Note that, unless stated otherwise, you cannot use `~` in any of the paths in this
file (you must write out the full path to your home directory, e.g.,
`/home/billbrod` or `/Users/billbrod`). Also, the paths should probably not have
capital letters -- there's a discrepancy between how Mac and Linux handle
capital letters in paths, which might create problems.

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
    participants before the experiment (see [Training](#training) section for
    how to use).
  - `download_data.py`: script to download and arrange data for reproducing
    results and figures. See [Download data](#download-data) for how to use.
  - `matlab/`: two matlab scripts using external matlab libraries. Neither are
    necessary: one is used to generate the windows from the Freeman and
    Simoncelli, 2011 paper (the output of which can be downloaded using
    `download_data.py`) and the other generates some LGN-like image statistics
    that we didn't end up using.
  - `data/`: contains some data files.
    - `Dacey1992_RGC.csv`: csv containing data from figure 2B of Dacey and
      Petersen, 1992, extracted using
      [WebPlotDigitizer](https://apps.automeris.io/wpd/) on July 15, 2021. To
      recreate that figure, use the snakemake rule `dacey_figure`. Note that we
      did not separate the data into nasal field and temporal, upper, and lower
      fields, as the paper does.
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
       [Usage](#usage) section for details on how to create them.
     - `figure_rules.txt`: this is a list of snakemake rules that create figures
       (rather than analyze the data). It can be used to limit snakemake's
       search of possible analysis paths. See [Usage](#usage) for details on how
       to use.
     - `pip_freeze_laptop.txt`, `pip_freeze_hpc.txt`: the outputs of `pip
       freeze` on two different machines, showing working environments as of
       fall 2022.
  - `tests/test_models.py`: contains a small number of tests of the pooling
    models, ran weekly and on every push (alongside other tests).
  - `environment-psychopy.yml`, `environment.yml`: yml files defining conda
    environment for the experiment (using `psychopy`) and for everything. See
    [Setup](#setup) section for how to use.
  - `greene.json`, `rusty.json`: json files defining how snakemake should
    communicate with NYU's and Flatiron's SLURM clusters, respectively (works
    with the [snakemake-slurm](https://github.com/billbrod/snakemake-slurm)
    profile). I have written a section on how to use snakemake with a SLURM
    cluster in a readme [for a different
    project](https://github.com/billbrod/spatial-frequency-preferences#cluster-usage),
    and may write something in more detail at some point. Reach out if you have
    questions.
  - `config.yml`: yml configuration file, defining paths, metamer path template,
    and some configuration for experiment structure, see [here](#configyml) for
    details.

# Usage details

The following sections contain some extra details that may be useful.

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
cached outputs directly. If you wish to run the notebook locally, you'll need to
download the input files (`python download_data.py freeman2011_check_input`) and
then either download the files required (`python download_data.py
freeman2011_check_output`) or generate them yourself (`snakemake -prk
freeman_check`). Note that generating them yourself will require MATLAB with the
[Freeman metamer](https://github.com/freeman-lab/metamers/) and
[matlabPyrTools](https://github.com/LabForComputationalVision/matlabPyrTools)
toolboxes (set their paths correctly in `config.yml`)

Make sure you have [Jupyter](#jupyter) installed, then navigate to the
`notebooks/` directory on your terminal and activate the environment you install
jupyter into (`metamers` or `base`, depending on how you installed it), then run
`jupyter` and open up the notebook. If you used the `nb_conda_kernels` method,
you should be prompted to select your kernel the first time you open a notebook:
select the one named "metamers".

A portion of the results presented in this notebook are also found in one of the
paper's appendices.

## Experiment notes

These are the notes I wrote for myself so I remembered how to run the experiment
and could teach others.

To run the experiment, make sure that the stimuli array and presentation indices
have been generated and are at the appropriate path. It's recommended that you
use a chin-rest or bite bar to guarantee that your subject remains fixated on
the center of the image; the results of the experiment rely very heavily on the
subject's and model's foveations being identical.

### Training

To teach the subject about the experiment, we want to introduce them to the
structure of the task and the images used. The first one probably only needs to
be done the first time a given subject is collecting data for each model /
comparison, the second should be done at the beginning of each session.

(The following paths all assume that `DATA_DIR` is `~/Desktop/metamers`, replace
that with your actual path.)

1. First, run a simple training run (if you haven't already, first run `python
   download_data.py experiment_training` to get the required files):
    - `conda activate psypy` 
    - `python foveated_metamers/experiment.py ~/Desktop/metamers/stimuli/training_noise/stimuli_comp-{comp}.npy sub-training 0 ; python foveated_metamers/experiment.py ~/Desktop/metamers/stimuli/training_{model}/stimuli_comp-{comp}.npy sub-training 0` 
       where `{comp}` depends on which comparison you're running, and `{model}`
       is `RGC_norm_gaussian` or `V1_norm_s6_gaussian`, depending on which
       you're running.
    - Explanatory text will appear on screen, answer any questions.
    - This will run two separate training runs, both about one or two minutes,
      with feedback between each trial (the fixation dot will turn green if they
      answered correctly) and at the end of the run (overall performance).
    - The first one will just be comparing natural to noise images and so the
      subject should get 100%. The goal of this one is to explain the basic
      structure of the experiment.
    - The second will have two metamers, one easy and one hard, for each of two
      reference images. They should get 100% on the easy one, and about chance
      on the hard. The goal of this one is to show what the task is like with
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
   - Note: this *will not work* unless you have the metamers for this comparison
     at the locations where they end up after synthesis (you probably don't). If
     you're interested in using this script, open an issue and I'll try to
     rework it.

### Task structure

Unlike previous experiments, we use a split-screen task. Each trial lasts 1.4
seconds and is structured like so:

```
|Image 1 | Blank  |Image 2 |Response|  Blank |
|--------|--------|--------|--------|--------|
|200 msec|500 msec|200 msec|        |500 msec|
```

Image 1 will consist of a single image divided vertically at the center by a
gray bar. One half of image 2 will be the same as image 1, and the other half
will have changed. The two images involved are either two metamers with the same
scaling value (if `comp=met` or `met-natural`) or a metamer and the reference
image it is based on (if `comp=ref` or `ref-natural`). The subject's task is to
say whether the left or the right half changed. They have as long as they need
to respond and receive no feedback.

To run the experiment:

- Activate the `psypy` environment: `conda activate psypy`
- Start the experiment script from the command line: 
   - `python foveated_metamers/experiment.py ~/Desktop/metamers/stimuli/{model}/stimuli_comp-{comp}.npy {subj_name} {sess_num}` 
     where `{model}, {subj_name}, {sess_num}, {comp}` are as described in the
     [training](#training) section.
   - There are several other arguments the experiment script can take,
     run `python foveated_metamers/experiment.py -h` to see them, and
     see the [other arguments](#other-arguments) section for more
     information.
- Explain the task to the subject, as given in the example below (similar text
  will also appear on screen before each run for the participant to read)
- When the subject is ready, press the space bar to begin the task.
- You can press the space bar at any point to pause it, but the pause won't
  happen until the end of the current trial, so don't press it a bunch of times
  because it doesn't seem to be working. However, try not to pause the
  experiment at all.
- You can press q/esc to quit, but don't do this unless truly necessary.
- There will be a break half-way through the block. The subject can get up,
  walk, and stretch during this period, but remind them to take no more than a
  minute. When they're ready to begin again, press the space bar to resume.
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

### Experiment Checklist

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
python foveated_metamers/experiment.py ~/Desktop/metamers/stimuli/training_noise/stimuli_comp-ref.npy sub-training 0 ; python foveated_metamers/experiment.py ~/Desktop/metamers/stimuli/training_{model}/stimuli_comp-ref.npy sub-training 0
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
python foveated_metamers/experiment.py ~/Desktop/metamers/stimuli/{model}/stimuli_comp-ref.npy {subj_name} {sess_num}
```

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
3. As of June 2023, there appears to be some issue with svgutils and inkscape,
   where any svg file that is the output of my `compose_figures` rule cannot be
   opened by inkscape (attempting to do so leads inkscape to immediately crash).
   This means that none of the files in the `compose_figures` directory can be
   used as inputs to the `embed_bitmaps_into_figures` rule (but those
   in`figures` can be). I'm unclear why this is happening now, but have been
   unable to track it down.
   
# Notes on reproducibility

The intention of sharing this code is to allow for the reproduction of the
figures in the resulting paper. This is the code I used to synthesize the
metamers used in the experiment, and you can use it to do so, but there are a
couple things you should be aware of:

- Results will not be identical on CPUs and GPUs. See PyTorch's
  [notes](https://pytorch.org/docs/stable/notes/randomness.html) on this.
- I used stochastic weight averaging (SWA) for the energy model metamers. SWA
  seems to reduce the final loss by averaging metamer pixel values as we get
  near convergence (see
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
and energy models used in this project. Feel free to use the models in this
repo, but the simpler version from that README may better suit your needs. The
version in this repo includes a bunch of helper code, including for creating
plots and the starts of paths not taken. The only important thing missing from
the `pooling-windows` repo is normalization -- look for the `normalize_dict`
attribute in `extra_packages/plenoptic_part/simulate/ventral_stream.py` to see
how I implemented that, and feel free to reach out if you have trouble.

You may also be interested in the code used by two other papers that this
project references a lot, [Freeman and Simoncelli,
2011](https://github.com/freeman-lab/metamers/) and [Wallis et al.,
2019](https://zenodo.org/record/2582880). If you wish to use the Freeman code,
note the bug in window creation pointed out by Wallis et al. (discussed in their
appendix 1).

# Citation

If you use the data, code, or stimuli from this project in an academic
publication, please cite the
[preprint](https://doi.org/10.1101/2023.05.18.541306). If you use the code,
please additionally cite the [zenodo
doi](https://zenodo.org/badge/latestdoi/195451020) for the corresponding release
(e.g., `v1.0-biorxiv` corresponds to the DOI
`https://doi.org/10.5281/zenodo.7948552`).

# References

- Dacey, D. M., & Petersen, M. R. (1992). Dendritic field size and morphology of
  midget and parasol ganglion cells of the human retina. Proceedings of the
  National Academy of Sciences, 89(20), 9666–9670.
  http://dx.doi.org/10.1073/pnas.89.20.9666

- Freeman, J., & Simoncelli, E. P. (2011). Metamers of the ventral
  stream. Nature Neuroscience, 14(9),
  1195–1201. http://dx.doi.org/10.1038/nn.2889

- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Paul-Christian
  B\"urkner (2021). Rank-normalization, folding, and localization: an improved
  $R$ for assessing convergence of mcmc (with discussion). Bayesian Analysis,
  16(2), 667–718. http://dx.doi.org/10.1214/20-BA1221

- Wallis, T. S., Funke, C. M., Ecker, A. S., Gatys, L. A., Wichmann, F. A., &
  Bethge, M. (2019). Image content is more important than bouma's law for scene
  metamers. eLife, 8(), . http://dx.doi.org/10.7554/elife.42512

- Wang, Z., & Simoncelli, E. P. (2008). Maximum differentiation (MAD)
  competition: A methodology for comparing computational models of perceptual
  discriminability. Journal of Vision, 8(12), 1–13.
  http://dx.doi.org/10.1167/8.12.8
