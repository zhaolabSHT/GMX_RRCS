# gmx_rrcs
A script can calculate residues-residues contact scores (rrcs) of a trajectory file created by GROMACs.


## Usage
执行脚本时，请使用以下命令查看帮助信息：
```bash
python gmx_rrcs.py --help
```
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
  --res_file RES_FILE   Path to residue information file
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
