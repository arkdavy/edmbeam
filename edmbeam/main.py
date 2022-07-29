import sys
import argparse
import importlib

from blochwave import bloch as bl
from .hw import calculate_hw


def argument_parser():
   parser = argparse.ArgumentParser()
   parser.add_argument('--cif', help='cif file path')
   parser.add_argument('--device', help='switch between "CPU" or "GPU", default is CPU. Works with Howie-Whelan only at the moment')
   parser.add_argument('--hw-input', help='input for HW calculations in python code format')
   parser.add_argument('--hw-outimage', help='output image file name with extension defining the format, default is "out.png"')
   parser.add_argument('--calculation', help='"HW" (Howie-Whelan) or "BW" (BlochWave). This part may change in the future')
   args, unknown_args = parser.parse_known_args()
   return args, unknown_args


def run_edmbeam(cif=None, inp=None, device=None, calculation=None, outimage=None):

   print("\n=== edmbeam code ===\n")
   print("... run 'edmbeam --help' to see available command-line options ...\n")

   # parse the command-line arguments
   args, unknown_args = argument_parser()
  
   if not len(sys.argv) > 1:
       raise ValueError("no arguments provided, give at least one")

   print("The following arguments have been provided:")
   for arg in vars(args):

       # value of a command-line argument
       val = getattr(args, arg)

       # print out the command line arguments and values
       if (val): print('  {:>20}: {}'.format(arg, val))


   # save the unput into variables
   for arg in vars(args):

       # value of a command-line argument
       val = getattr(args, arg)

       # re-init the config class with a new default configuration
       if arg == 'cif' and val: 
          cif = val

       if arg == 'device':
          device = val

       if arg=='hw_input' and val:
          spec = importlib.util.spec_from_file_location('input', val)
          inp = importlib.util.module_from_spec(spec)
          sys.modules['input'] = inp
          spec.loader.exec_module(inp)

       if arg=='hw_outimage' and val:
          outimage = val

       if arg=='calculation' and val:
          calculation = val

   if inp is None: raise ValueError("missing the input file for Howie-Wehlan calculations (--hw-input flag)")
   if device is None: device = 'CPU'
   if outimage is None: outimage = 'out.png'
   if calculation is None: calculation = 'HW'

   if calculation == 'HW':
      print("\n... running HW code ...\n")
      calculate_hw(device, inp, outimage)
      print("\n... HW calculation finished ...\n")

   elif calculation == 'BW':
       if cif is None: raise ValueError("missing the cif file for the blochwave code (--cif flag)")
       print("... calling functions from Blochwave ...\n")
       b0 = bl.Bloch(args.cif,path='',u=[0,0,1],Nmax=3,Smax=0.1,opts='svt')
       b0.show_beams_vs_thickness()
       b0.convert2tiff()
       print("\n... BlochWave calculation finished...\n")

   else:
       raise ValueError('unrecognised "calculation" option')


if __name__ == "__main__":
    run_edmbeam()

