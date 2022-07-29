import setuptools

setuptools.setup(
    name = "edmbeam",
    author = "Oliver Bunting, Arkadiy Davydov, Richard Beanland",
    version = '0.0.1',
    author_email="Oliver.Bunting@warwick.ac.uk, arkadiy.davydov@warwick.ac.uk, R.Beanland@warwick.ac.uk",
    description = "Basic transmission electron microscope image simulations of disloctions",
    # Dependencies/Other modules required for your package to work
    install_requires=[
      'wheel',
      'tqdm',
      'pycairo',
      'PyGObject',
      'argparse',
      'opencv-python',
      'cupy-cuda116',
      'ccp4ed @ git+https://github.com/arkdavy/electron-diffraction.git@exp#egg=ccp4ed'
    ],
 #   package_data={'':['']},
    include_package_data=True,
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'edmbeam = edmbeam.main:run_edmbeam',
            'edmbeam_showtif = edmbeam.showtif:showtif',
            'edmbeam_converttif = edmbeam.converttif:converttif',
        ],
    },
    classifiers = [
        "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
    ],
)
