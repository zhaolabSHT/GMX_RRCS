# GMX_RRCS

## Introduction
The gmx_rrcs script is designed to calculate the residues-residues contact scores (rrcs) from a trajectory file generated by GROMACS.


## Install
```bash
pip install gmx-rrcs
```

## Usage
For help information, execute the script with the following command:
```bash
gmx_rrcs --help
```
Here is the usage and argument configuration for the gmx_rrcs.py script:
```text
usage: gmx_rrcs.py [-h] --top_file TOP_FILE --traj_file TRAJ_FILE [--res_file RES_FILE] [--radius_min RADIUS_MIN]
                   [--radius_max RADIUS_MAX] [--output_dir OUTPUT_DIR] [--output_file OUTPUT_FILE]
                   [--begin_time BEGIN_TIME] [--end_time END_TIME] [--freq_step FREQ_STEP] [--plot]
                   [--filter_threshold FILTER_THRESHOLD] [--num_processes NUM_PROCESSES] [--print_freq PRINT_FREQ]

Process configuration parameters.

optional arguments:
  -h, --help            show this help message and exit
  --top_file TOP_FILE   Topology file path (required)
  --traj_file TRAJ_FILE
                        Trajectory file path (required)
  --res_file RES_FILE   Path to residue information filePath to the file containing residue pair indices.
  --radius_min RADIUS_MIN
                        Minimum distance threshold in Ångström, default is 3.23
  --radius_max RADIUS_MAX
                        Maximum distance threshold in Ångström, default is 4.63
  --output_dir OUTPUT_DIR
                        Directory path where output files will be saved. If not specified, the current directory will be   
                        used.
  --output_file OUTPUT_FILE
                        Output file name, default is 'RRCS_output.txt'
  --begin_time BEGIN_TIME
                        Start time for calculation in picoseconds, default is 0.0
  --end_time END_TIME   End time for calculation in picoseconds, default is 9999999.0
  --freq_step FREQ_STEP
                        Time step for analysis, default is 0.1 ps
  --plot                Generate a plot if specified (default: False).
  --filter_threshold FILTER_THRESHOLD
                        Choose whether to output the high-scoring results that have been filtered, default is 3.0.
  --num_processes NUM_PROCESSES
                        Number of processes for parallel execution.If None, use all available CPU cores. Default is None.  
  --print_freq PRINT_FREQ
                        Print the elapsed time every N frames, default is 1000 frames.
```

## Example Usage
```bash
gmx_rrcs --top_file your_topology_file --traj_file your_trajectory_file --res_file your_residue_pair_indices_file  --output_file your_output_file --output_file your_output_dir
```


## File Formats
### Topology and Trajectory Files
We use the MDAnalysis library to read and process GROMACS trajectories. Therefore, the --top_file and --traj_file parameters accept file formats supported by MDAnalysis. Ensure that your input files conform to MDAnalysis format requirements. In our tests, there have been occasional issues with reading .tpr files, so we recommend using .pdb or .gro files as the input topology files for the --top_file parameter. For trajectory files, we recommend using the .xtc format. It is crucial to check the completeness and correctness of your protein and trajectory in the input files before running gmx_rrcs to ensure accurate results.

### Residue Pair Indices File
This file format is custom-defined. Each line specifies a pair of residue indices separated by the $ symbol, where the content before $ is the first member, and the content after $ is the second member. A simple example is as follows:
```
15 $ 28         ; This line defines a residue pair (15, 28), where the first member is residue 15 and the second member is residue 28.
```
The content after the ; character is considered a comment and will be ignored during parsing.

Multiple residue pairs can be specified on the same line by separating them with spaces:
```
15 16 $ 28      ; This line defines two residue pairs: (15, 28) and (16, 28).
```

Similarly, multiple residues can be specified for the second member:
```
15 $ 28 29      ; This line defines residue pairs (15, 28) and (15, 29).
```

Both members can specify multiple residues:
```
15 16 $ 28 29   ; This line defines residue pairs (15, 28), (15, 29), (16, 28), and (16, 29).
```

You can also specify a range of residues using the - symbol:
```
15-17 20 $ 28   ; This line defines residue pairs (15, 28), (16, 28), (17, 28), and (20, 28).
```

If a line does not contain the $ symbol, the residues on that line will pair with each other:
```
15 28 40        ; This line defines residue pairs (15, 28), (15, 40), and (28, 40).
```

The residue pair indices file allows multiple lines to define residue pairs:
```
15 $ 28
32 35 $ 10
46 $ 55 16
78 58 $ 98 61
99-102 $ 293-299
```
Regardless of the number of lines, they will all be merged into a single list of residue pairs.
For GMX_RRCS, the residue pair indices file is not a required parameter. If you do not provide this file, GMX_RRCS will automatically generate a list of all possible residue pairs in the protein, which will significantly increase computation time. Hence, it is generally not recommended.

