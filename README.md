# foveated-metamers

Create metamers using models of the ventral stream and run experiments to validate them

This project starts with a replication of Freeman and Simoncelli,
2011, out to higher eccentricities, and will extend it by looking at
spatial frequency information as well.

# Dockerfile

In order to build Dockerfile, have this directory and the most recent
version of `plenoptic` in the same directory and then FROM THAT
DIRECTORY (the one above this one), run `sudo docker build
--tag=foveated-metamers -f foveated-metamers/Dockerfile .`. This
ensures that we can copy plenoptic over into the Docker container.

Once we get plenoptic up on pip (or even make it public on github), we
won't need to do this. At that time, make sure to replace
`foveated-metamers/environment.yml` with `environment.yml` and remove
the plenoptic bit.

# Requirements

Need to make sure you have ffmpeg on your path when creating the
metamers, so make sure it's installed.

When running on NYU's prince cluster, can use `module load
ffmpeg/intel/3.2.2` or, if `module` isn't working (like when using the
`fish` shell), just add it to your path manually (e.g., on fish: `set
-x PATH /share/apps/ffmpeg/3.2.2/intel/bin $PATH`)

For running the experiment, need to install `glfw` from your package
manager.

# References

- Freeman, J., & Simoncelli, E. P. (2011). Metamers of the ventral
  stream. Nature Neuroscience, 14(9),
  1195–1201. http://dx.doi.org/10.1038/nn.2889
