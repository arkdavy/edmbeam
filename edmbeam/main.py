import sys
import argparse
from blochwave import bloch as bl

def argument_parser():
   parser = argparse.ArgumentParser()
   parser.add_argument('--cif', help='cif file path')
   parser.add_argument('--temp-flag', help='tempoary flag comma-separated pattern list of files to be copied from the temporary directory')
   args, unknown_args = parser.parse_known_args()
   return args, unknown_args

def run_ed():

   print("\n... Entered run_ed() ...\n")
   print("\n... run 'edmbeam --help' to see available command-line options ...\n")

   # parse the command-line arguments
   args, unknown_args = argument_parser()

   print("The following arguments have been provided:")
   for arg in vars(args):

       # value of a command-line argument
       val = getattr(args, arg)

       # print out the command line arguments and values
       if (val): print('  {:>20}: {}'.format(arg, val))

   print("\n")
   print("Matching flags in the code:")
   for arg in vars(args):

       # value of a command-line argument
       val = getattr(args, arg)

       # re-init the config class with a new default configuration
       if (arg=='cif' and val): 
          print("-Matched cif")

       # re-init the config class with a new default configuration
       if (arg=='temp_flag' and val): 
          print("-Matched temp_flag")
          print("   note that temp-flag is stored and matched as temp_flag in the code") 
          print("   check also this printing command, which uses its alternative access, {}".format(args.temp_flag))

   print("\n")

   print("calling functions from Blochwave")
   
   b0 = bl.Bloch(args.cif,path='',u=[0,0,1],Nmax=3,Smax=0.1,opts='svt')
   b0.show_beams_vs_thickness()
   b0.convert2tiff()

if __name__ == "__main__":
    run_ed()

