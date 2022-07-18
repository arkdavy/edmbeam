# hw2beam

Basic transmission electron microscope image simulations of disloctions (diffraction contrast) based on the 2-beam Howie-Whelan equations. 
The core calculations can be performed on either CPU or GPU dictated by the `--device` flag. Certain functions rely on the 
[electron diffraction](https://github.com/ccp4/electron-diffraction) code developed by Tarik Drevon (STFC).

Run `pip install .` to install the package (or `pip install . --user` for the `home/` installation)

We have found two minor problems during installation on *Ubuntu*
1) If the python wheel package is not installed beforehand, the `invalid command 'bdist_wheel'` error appears. Running the setup command above the second time resolves the issue
2) If the `PyGObject` dependency needs to be installed (this may change in future versions of the software), addintional `libgirepository1.0-dev` system package has to be installed as well
