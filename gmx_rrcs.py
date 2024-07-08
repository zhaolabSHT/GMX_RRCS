# -*- coding: utf-8 -*-

import os
import itertools
import logging
import argparse

import timeit
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from numba import jit

from collections import defaultdict
from typing import Dict, Union
from termcolor import colored 
from MDAnalysis import Universe

from utilities import (
    log_error, log_warning, timing_decorator,
    InputFileError, ParameterWrongError,
    ResidueIndexError, # compute_frame_rrcs
)
from constants import (
    THREE_TO_ONE_LETTER, MAIN_CHAINS_ATOMS, 
    ATOM_DISTANCE_THRESHOLD, OUTPUT_OFFSET,
    MAX_INDEX_DIFFERENCE_FOR_NEIGHBORS,
    OUTPUT_HEADER, OUTPUT_FILTER_HEADER,
)

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ConfigParser:
    def __init__(self):
        """config arguments from command line"""
        self.parser = argparse.ArgumentParser(description="Process configuration parameters.")
        
        # define required parameter
        self.parser.add_argument('--top_file', type=str, required=True, 
                                 help="Topology file path (required)")
        self.parser.add_argument('--traj_file', type=str, required=True, 
                                 help="Trajectory file path (required)")
        
        # define optional parameter
        self.parser.add_argument('--res_file', type=str, default='',
                                 help="Path to residue information file")
        self.parser.add_argument('--radius_min', type=float, default=3.23,
                                 help="Minimum distance threshold in Ångström, default is 3.23")
        self.parser.add_argument('--radius_max', type=float, default=4.63,
                                 help="Maximum distance threshold in Ångström, default is 4.63")
        self.parser.add_argument('--output_dir', type=str, default='',
                                 help="Directory path where output files will be saved. If not specified, the current directory will be used.")
        self.parser.add_argument('--output_file', type=str, default='RRCS_output.txt',
                                 help="Output file name, default is 'RRCS_output.txt'")
        self.parser.add_argument('--begin_time', type=float, default=0.0,
                                 help="Start time for calculation in picoseconds, default is 0.0")
        self.parser.add_argument('--end_time', type=float, default=9999999.0,
                                 help="End time for calculation in picoseconds, default is 9999999.0")
        self.parser.add_argument('--freq_step', type=float, default=0.1,
                                 help="Time step for analysis, default is 0.1 ps")
        self.parser.add_argument('--plot', action='store_true',
                                 help='Generate a plot if specified (default: False).')
        self.parser.add_argument('--filter_threshold', type=float, default=3.0,
                                 help='Choose whether to output the high-scoring results that have been filtered, default is 3.0.')


    def parse_arguments(self) -> Dict[str, Union[float, str, bool, int]]:
        """Parses command line arguments and returns them as a dictionary."""
        args = self.parser.parse_args()
        return vars(args)


class ResidueCombinePairs:
    def __init__(self, basic_settings):
        """
        Initialize a ResidueCombinePairs object.
        
        Parameters:
            basic_settings (dict): A dictionary containing basic settings.
        """
        self.basic_settings = basic_settings
        self.res_pair_set = set()

    def read_file(self):
        """Read and parse the index file, populate res_pair_set."""
        res_file = self.basic_settings["res_file"]
        if os.path.exists(res_file):
            with open(res_file, 'r') as f:
                lines = f.readlines()
            lines = self.clear_lines(lines)
            if len(lines) == 0:
                log_warning(
                    "InputFileWarning", 
                    "The ResidueIndex file is empty. All residue pairs will be processed, which may take a considerable amount of time."
                    )
                self.res_pair_set = self.combine_all_residues()
            else:
                for line in lines:
                    self.res_pair_set.update(self.parse_line(line))
        else:
            log_warning(
                "InputFileWarning", 
                "The ResidueIndex file does not exist. All residue pairs will be processed, which may take a considerable amount of time."
                )
            self.res_pair_set = self.combine_all_residues()
            
    def clear_lines(self, lines):
        """Clear the lines in the ResidueIndex file."""
        new_lines = []
        for line in lines:
            line = line.strip()
            if line:  # Check if line is not empty
                new_lines.append(line)
        return new_lines

    def parse_line(self, line):
        """
        Uniformly parse a line, handling both sides of the '$' symbol similarly.
        
        Parameters:
            line (str): The line to be parsed.
            
        Returns:
            tuple: The result of the parsing.
        """
        line = line.strip().split(";")[0].strip() # Remove comments
        if '$' in line:
            return self.parse_stand_line(line)
        else:
            return self.parse_other_line(line)
        
    def parse_stand_line(self, line):
        """Parse lines containing '$', generating all possible combinations of residue pairs."""
        res_pair_list = []
        parts = line.split("$")
        for part in parts:
            part = part.strip()  # Remove leading/trailing whitespaces
            if part:  # Check if part is not empty
                res_selc = self._parse_res_list(part, self.basic_settings['res_num'])
                res_pair_list.append(res_selc)
            else:
                raise ResidueIndexError
        assert len(res_pair_list) == 2, ResidueIndexError
        return set(itertools.product(*res_pair_list))
    
    def parse_other_line(self, line):
        """Parse lines not containing with '$', generating all possible combinations of residue pairs."""
        res_selc = self._parse_res_list(line.strip(), self.basic_settings['res_num'])
        return set(itertools.combinations(sorted(res_selc), 2))

    @staticmethod
    def _parse_res_list(res_list_str, res_num):
        """Parse a residue list from a string, supporting range notation."""
        res_selc = []
        for res in res_list_str.split():
            if '-' in res:
                start, end = map(int, res.split('-'))
                res_selc.extend(range(start, end + 1))
            else:
                res_selc.append(int(res))
        if 'all' in res_list_str:
            res_selc = list(range(1, res_num + 1))
        return set(res_selc)
    
    def combine_all_residues(self):
        """
        Generates all possible residue pairs from an MD trajectory.

        This function checks if the MD trajectory has a 'residues' attribute,
        extracts the residue IDs, sorts them if necessary, and then generates
        all unique pairs of consecutive residues. It handles edge cases such as
        missing keys or attributes in the input data and catches any unexpected
        exceptions to ensure robustness.

        Returns:
            generator: A generator that yields tuples representing each pair of
                    consecutive residues in the sorted list.

        Raises:
            KeyError: If 'md_traj' key is missing in the basic settings.
            AttributeError: If 'residues' attribute is missing in the MD trajectory.
            Exception: For any other unexpected errors during execution.
        """
        md_traj = self.basic_settings['md_traj']
        # Extract resid values from residues
        residues = [res.resid for res in md_traj.residues]
        return set(itertools.combinations(residues, r=2))

    def get_res_pairs(self):
        """Retrieve the parsed set of residue pairs."""
        pairs = tuple(sorted(self.res_pair_set))
        logging.info(f"Read {len(pairs)} residue pairs.")
        return pairs

@timing_decorator
class UniverseInitializer:
    def __init__(self, basic: dict):
        self.basic = basic

    def check_radius(self):
        """
        Ensure the radius minimum and maximum values are within a valid range.
        
        This method adjusts the minimum and maximum radius values in the basic configuration to ensure they are not less than 0.
        If the minimum radius is less than 0, it is set to 0; if the maximum radius is less than the minimum, the maximum is then set to the minimum.
        This process guarantees the reasonableness of radius settings, avoiding invalid or unreasonable search ranges.
        
        Returns:
            No return value, but modifies the 'radius_min' and 'radius_max' values in the self.basic dictionary.
        """
        # Ensure the radius minimum is non-negative
        self.basic['radius_min'] = max(self.basic['radius_min'], 0)
        self.basic['radius_max'] = max(self.basic['radius_min'], self.basic['radius_max'])

    def check_file_exists(self):
        """
        Verifies the existence of required input files.
        
        This method iterates through a list of file paths, checking if each file exists at the corresponding path. 
        If a file is not found, an exception is raised.
        """

        # Iterates through the file path list
        for file_type in ['top_file', 'traj_file']:
            # Checks if the file exists, and raises an InputFileError if not
            if not os.path.exists(self.basic[file_type]):
                raise InputFileError(self.basic[file_type])

    def initialize_universe(self):
        """
        Initialize the Universe object.

        This method creates an MDAnalysis Universe object. It uses the topology file and trajectory file paths retrieved from the 
        `basic` dictionary to instantiate the Universe. Upon initialization, it stores the number of residues in the 
        universe back into the `basic` dictionary for future reference.

        Returns:
            None
        """
        # Instantiate the Universe object with topology and trajectory file paths
        self.basic['md_traj'] = Universe(self.basic['top_file'], self.basic['traj_file'])
        # Store the number of residues in the universe into the basic dictionary
        self.basic['res_num'] = len(self.basic['md_traj'].residues)

        parser = ResidueCombinePairs(self.basic)
        parser.read_file()
        self.basic["res_pairs"] = parser.get_res_pairs()

    def calculate_time_min(self):
        """
        Calculates the minimum time step for the trajectory.

        This method computes the average time difference between trajectory points 
        by taking the total time span from the first to the last point and dividing 
        it by the number of intervals, which is one less than the number of points.
        The time step is a measure of the interval between successive trajectory points,
        useful for understanding the rate of motion or change.

        Note: This method takes no explicit parameters and does not return a value;
        instead, it updates the object's state by setting the instance dict `basic`.
        """
        # Compute the time difference and divide by the number of intervals to get the mean time step
        self.basic["time_resolution_min"] = int((self.basic['md_traj'].trajectory[-1].time - self.basic['md_traj'].trajectory[0].time) / (len(self.basic['md_traj'].trajectory) - 1))

    def check_time_interval(self):
        """
        Validates the time interval settings to ensure proper configuration.

        This method ensures that the time interval is within valid bounds and does not lead to calculation errors. Specifically, it:
        1. Guarantees the start time is not less than 0.
        2. Ensures the end time does not exceed the last moment of the entire trajectory.
        3. Confirms the time step is not less than the minimum allowed time step.
        4. Throws an exception if the time step is greater than the end time, as this would inhibit meaningful computations.

        Parameters:
        self: The instance of the class, holding basic configuration and status information.

        Raises:
        ParameterWrongError: If the time interval is incorrectly set, this exception is thrown.
        """
        # Ensure the start time is non-negative
        self.basic['begin_time'] = max(self.basic['begin_time'], 0.0)
        # Ensure the end time does not surpass the final moment of the trajectory
        self.basic['end_time'] = min(self.basic['end_time'], self.basic['md_traj'].trajectory[-1].time)
        # Ensure the time step is at least the minimum time step
        self.basic['freq_step'] = max(self.basic["time_resolution_min"], self.basic['freq_step'])
        # Throw an exception if the time step exceeds the end time
        if self.basic['freq_step'] > self.basic['end_time']:
            raise ParameterWrongError('--freq_step')

    def get_chain(self):
        """
        Retrieves the chain information from the trajectory.
        """
        chains = []
        for chain in self.basic["md_traj"].segments:
            chains.append((chain.segindex, chain.segid))
        self.basic['traj_chains'] = chains

    def check(self):
        """
        Conducts a series of checks to ensure all prerequisites for the program's execution are met.
        
        This method invokes multiple verification functions to respectively confirm the availability of the plotting library,
        the reasonableness of the set radius, the existence of the RRCs file, the presence of required files, 
        the initialization of the universe model, the appropriateness of the minimum time step, and the validity of the time interval settings.
        These checks guarantee that the program operates under the correct configurations and conditions, averting potential errors and exceptions.
        """
        
        # Check if the specified radius is reasonable
        self.check_radius()
        # Verify the existence of necessary files
        self.check_file_exists()
        # Initialize the universe model
        self.initialize_universe()
        # Compute the minimum time step
        self.calculate_time_min()
        # Validate the time interval configuration
        self.check_time_interval()
        # Get chains from trajectory
        self.get_chain()


@timing_decorator
class ResidueContactAnalyzer:
    """
    A class to analyze residue-residue contacts in molecular dynamics simulations.
    """
    
    def __init__(self, basic_settings):
        """
        Initialize the analyzer with the path to the simulation universe and basic settings.
        
        Parameters:
        - basic_settings (dict): Dictionary containing settings like begin_time, end_time, freq_step, radius_min, radius_max.
        """
        self.basic_settings = basic_settings
        self.all_frame_rrcs = {}
        self.member_first = []
        self.member_second = []
        
    def _load_residues(self):
        """
        Processes a given list of tuples, extracting the first and second elements 
        from each tuple into two separate sets to eliminate duplicates. Finally, 
        it returns two sorted lists containing these unique elements.

        Args:
        - res_pairs: A list containing tuples. The first and second elements from 
                    each tuple will be extracted.

        Returns:
        - Two sorted lists containing all unique first and second elements from 
        the tuples, respectively.
        """
        res_pairs = self.basic_settings['res_pairs']

        # Input validation
        if not isinstance(res_pairs, tuple):
            logging.error("res_pairs must be a tuple")
        if not all(isinstance(pair, tuple) and len(pair) == 2 for pair in res_pairs):
            logging.error("Each element in res_pairs must be a tuple of length 2")
        
        member_first = set()
        member_second = set()
        
        for pair in res_pairs:
            member_first.add(pair[0])
            member_second.add(pair[1])
        self.member_first = sorted(member_first)
        self.member_second = sorted(member_second)
    
    @staticmethod
    @jit(nopython=True)
    def _compute_rrcs_jit(distances_matrix, d_max_squared, d_min_squared):
        """
        JIT-compiled function to calculate RRCS more efficiently.
        
        Parameters:
        - distances_matrix (np.ndarray): Matrix of squared distances.
        - d_max_squared (float): Squared maximum distance for contact consideration.
        - d_min_squared (float): Squared minimum distance for full score.
        
        Returns:
        - total_score (float): The computed RRCS.
        """
        # Apply conditions using NumPy's where function
        scores = np.where(
            distances_matrix >= d_max_squared,
            0.0,
            np.where(
                distances_matrix <= d_min_squared,
                1.0,
                1.0 - ((distances_matrix - d_min_squared) / (d_max_squared - d_min_squared))
            )
        )

        # Sum all elements in the score matrix
        return np.sum(scores)
    
    def analyze_contacts(self):
        """
        Main function to iterate through trajectory frames, calculate residue-residue contacts, and their scores.
        """
        global_start = timeit.default_timer()
        md_traj = self.basic_settings['md_traj']
        
        begin_time_index = int(self.basic_settings['begin_time'] / self.basic_settings['time_resolution_min'])
        end_time_index = int(self.basic_settings['end_time'] / self.basic_settings['time_resolution_min'])
        frequency_step_index = int(self.basic_settings['freq_step'] / self.basic_settings['time_resolution_min'])

        for frame_step in md_traj.trajectory[begin_time_index:end_time_index+1:frequency_step_index]:
            frame_count = frame_step.frame + 1
            if frame_count % 50000 == 0:
                elapsed = timeit.default_timer() - global_start
                logging.info(f"Calculated to frame {frame_count}, took {elapsed:.2f} seconds") 
            
            info_first = self._get_residue_info(self.member_first)
            info_second = self._get_residue_info(self.member_second)
            frame_rrcs = []
            for chain_ix, chain_id in info_first.keys():
                info_res_first = info_first[(chain_ix, chain_id)]
                info_res_second = info_second[(chain_ix, chain_id)]
                for index_i, index_j in self.basic_settings['res_pairs']:
                    res_i = self.get_residue_name(md_traj, index_i, chain_ix)
                    res_j = self.get_residue_name(md_traj, index_j, chain_ix)
                    info_i = info_res_first[f"{index_i}{res_i}"]
                    info_j = info_res_second[f"{index_j}{res_j}"]
                    is_adjacent = self.are_residues_adjacent(index_i, index_j)
                    coord_i = self.adjest_atom_coordinates(is_adjacent, info_i)
                    coord_j = self.adjest_atom_coordinates(is_adjacent, info_j)
                    if self.prefilter_contacts(coord_i, coord_j):
                        dist = self.get_distances(coord_i, coord_j)
                        radius_min = self.basic_settings['radius_min']
                        radius_max = self.basic_settings['radius_max']
                        rrcs_score = self._compute_rrcs_jit(dist, radius_max, radius_min)
                    else:
                        rrcs_score = 0
                    frame_rrcs.append((f"{chain_id}:{index_i}{res_i}", f"{chain_id}:{index_j}{res_j}", rrcs_score))
            self.all_frame_rrcs[frame_count] = frame_rrcs


    @staticmethod
    # @jit(nopython=True)
    def prefilter_contacts(coord_i, coord_j):
        """
        Pre-filter the contacts to reduce the number of calculations.
        
        Parameters:
        coord_i, coord_j: Numpy arrays representing coordinates. Both are expected to have shape (n, 3),
                          where n is the number of coordinates.
        
        Returns:
        Boolean value indicating if there is at least one pair of coordinates closer than 4.14 units.
        """
        # Check if coord_i and coord_j are empty
        if not coord_i.size or not coord_j.size:
            return False
        
        # Compute coordinate differences and filter pairs closer than DISTANCE_THRESHOLD
        # coord_i shape: (m, 3) -> (m, 1, 3)
        # coord_j shape: (n, 3) -> (1, n, 3)
        # diff shape: (m, n, 3)
        diff = np.abs(coord_i[:, np.newaxis, :] - coord_j[np.newaxis, :, :])

        return np.any(np.prod(diff < ATOM_DISTANCE_THRESHOLD, axis=2))

    @staticmethod
    @jit(nopython=True)
    def get_distances(coord_i, coord_j):
        """
        Calculate the distances between two sets of coordinates using scipy's cdist function.
        
        Args:
            coord_i: A 2D numpy array representing the first set of coordinates. Each row is a coordinate.
            coord_j: A 2D numpy array representing the second set of coordinates. Each row is a coordinate.
        
        Returns:
            A 2D numpy array where element (i,j) represents the distance between coord_i[i] and coord_j[j].
        """
        diff = coord_i[:, np.newaxis, :] - coord_j[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))

    def adjest_atom_coordinates(self, is_neighbor, residues):
        """
        Adjust the coordinates of the atoms in the residue.
        
        Parameters:
        is_neighbor (bool): Whether the residues are neighbors.
        residues (list): List of tuples, each containing:
                    - atom_name (str): Name of the atom.
                    - atom_coord (np.array): Coordinate array [x, y, z].
                    - atom_occupancy (float): Occupancy of the atom.
        
        Returns:
        numpy.ndarray: Adjusted coordinates of the atoms.
        """
        try:
            adjest_coord = [
                atom_coord * atom_occupancy
                for atom_name, atom_coord, atom_occupancy in residues
                if not (is_neighbor and atom_name in MAIN_CHAINS_ATOMS)
            ]
            return np.array(adjest_coord)
        except (TypeError, ValueError) as e:
            log_warning(e, "Error processing atom coordinates")
            return np.array([])


    def are_residues_adjacent(self, index_i, index_j):
        """
        Check if two residues are neighbors within a specified distance.
        
        Parameters:
        - index_i: Integer representing the index of the first residue.
        - index_j: Integer representing the index of the second residue.
        - distance_threshold: The maximum distance (in Angstroms) considered as 'neighbor'. Default is 5A.
        
        Returns:
        - Boolean: True if the residues are neighbors within the specified distance, False otherwise.
        """
        return abs(index_i - index_j) < MAX_INDEX_DIFFERENCE_FOR_NEIGHBORS

    def get_residue_name(self, md_traj, index_i, chain_ix):
        """
        Get residue names from the universe.
        """
        atom = list(md_traj.select_atoms(f"resid {index_i} and not name H* and segindex {chain_ix}").ids - 1)[0]
        res_name = THREE_TO_ONE_LETTER.get(md_traj.atoms[atom].resname, 'X')
        return res_name

    def _get_residue_info(self, residues):
        """
        Retrieve residue information from the universe.
        
        Parameters:
        residues (list): List of residue IDs.
        
        Returns:
        - A dictionary containing residue information.
        """
        md_traj = self.basic_settings['md_traj']
        chains = self.basic_settings['traj_chains']
        pair_chain = defaultdict(dict)
        for _ix, _id in chains:
            _id = 'A' if _id == 'SYSTEM' else _id
            pair_residue = defaultdict(dict)
            for resid in residues:
                atom_ids = list(md_traj.select_atoms(f"resid {resid} and not name H* and segindex {_ix}").ids - 1)
                pair_atom = []
                for atom_id in atom_ids:
                    atom = md_traj.atoms[atom_id]
                    atom_name = atom.name
                    atom_coor = atom.position
                    atom_occu = atom.occupancy
                    res_name = atom.resname
                    pair_atom.append((atom_name, atom_coor, atom_occu))
                res_name = THREE_TO_ONE_LETTER.get(res_name, 'X')
                pair_residue[f'{resid}{res_name}'] = pair_atom
            pair_chain[(_ix, _id)] = pair_residue
        return pair_chain


@timing_decorator
class DataVisualizer:
    def __init__(self, basic_settings, rrcs_data):
        """
        Initializes the class with basic settings.

        This constructor takes a dictionary of basic settings, most importantly 
        the path to the output directory. If the output directory does not exist,
        it will be created.

        :param basic_settings: A dictionary containing basic settings, must include 
                            the path to the output directory.
               rrcs_data: A dictionary containing the RRCs data.
        """
        # Extract the output directory path from the basic settings
        self.output_dir = basic_settings['output_dir']
        
        # Check if the output directory exists, if not, create it
        if self.output_dir:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        else:
            self.output_dir = os.getcwd()
        
        # Concatenate the output file path based on the output directory path
        self.output = os.path.join(self.output_dir, basic_settings['output_file'])
        # Save whether to generate plots from the settings
        self.isplot = basic_settings['plot']
        self.rrcs_data = rrcs_data
        self.filter_threshold = basic_settings['filter_threshold']

    def write_output(self):
        """Writes the processed data to the output file."""
        # Initialize a list with the output file's header information.
        outlines = [OUTPUT_HEADER]
        filter_outlines = [OUTPUT_FILTER_HEADER]

        # Append the reformatted data lines.
        lines, filter_lines = self.reformat_data_lines(self.rrcs_data)

        outlines.extend(lines)
        with open(self.output, 'w') as f:
            f.writelines(outlines)
        logging.info(f"RRCS data is saved to {self.output} file.")

        # Construct the file path for the filtered data.
        filepath = f"{os.path.splitext(self.output)[0]}_filter_rrcs_greater_than_{self.filter_threshold}.txt"
        filter_outlines.extend(filter_lines)
        with open(filepath, 'w') as f:
            f.writelines(filter_outlines)
        logging.info(f"Filtered RRCS data is saved to the {filepath} file.")

    def reformat_data_lines(self, rrcs_data):
        """
        Reformat data lines to create a uniform representation for each pair of residues and their corresponding 
        Relative Rotamer Conformation Score (RRCS).
        
        This method starts by adding a header, then transforms the data within the rrcs_data dictionary into a 
        standardized list format, and finally prints these data in a neat table format through the pretty_print_table 
        method.
        
        Returns:
            Formatted data table, returned as a string.
        """
        outlines = [('Frame', 'Residue1', 'Residue2', 'RRCS')]
        filter_outlines = [('Frame', 'Residue1', 'Residue2', 'RRCS')]
        for frame, rrcs_list in rrcs_data.items():
            # Iterate over the RRCS list, each entry is a tuple containing residue identifiers and an RRCS score
            for res1, res2, rrcs_score in rrcs_list:
                outlines.append((frame, res1, res2, rrcs_score))
                if rrcs_score >= self.filter_threshold:
                    filter_outlines.append((frame, res1, res2, rrcs_score))
        outlines = self.pretty_print_table(outlines)
        filter_outlines = self.pretty_print_table(filter_outlines)

        return outlines, filter_outlines

    def pretty_print_table(self, rows):
        """
        Prints a table in a pretty format with specified column widths.
        
        :param rows: A list of lists where each inner list represents a row in the table.
        :param output_offset: An integer representing the offset to add to each column's width.
        :return: A string representation of the formatted table.
        """
        # Calculate column widths
        col_widths = [max(len(str(item))+OUTPUT_OFFSET for item in col) for col in zip(*rows)]
        # Builds the formatted table string based on the calculated column widths.
        table = [
            ''.join(str(item).ljust(width) for item, width in zip(row, col_widths))
            for row in rows
        ]
        return '\n'.join(table)

    def plot_scatter(self):
        """
        Plots an RRCS Scatter Diagram.

        Reformats data from self.rrcs_data and creates a scatter plot using the seaborn library.
        The scatter plot represents RRCS values for each frame, aiding in analyzing trends in RRCS values.

        The filename is derived from self.output and the plot is saved in PNG format.
        """
        # Reformatted scatter plot data
        x, y = self.reformat_scatter_data(self.rrcs_data)

        filename = f"{os.path.splitext(self.output)[0]}_scatter.png"

        # Chart title, x-axis and y-axis labels
        title = "RRCS scatter plot"
        xlabel = "Frame"
        ylabel = "RRCS Score"

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))

        sns.scatterplot(x=x, y=y, s=100, edgecolor='w', linewidth=0.5)

        # Set chart title, x-axis and y-axis labels text and styling
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)

        sns.despine()

        plt.grid(True, linestyle='--', alpha=0.7)

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logging.info(f"Scatter plot saved to {filename}")

    def plot_violin(self):
        """
        Generate a violin plot for the RRCSS scores.
        
        This method re-formats the RRCSS data, creates a violin plot using seaborn, and saves the plot as an image file.
        The violin plot shows the distribution of scores for the initial, middle, and final frames.
        """
        # Reformatted scatter plot data
        data = self.reformat_violin_data(self.rrcs_data)

        filename = f"{os.path.splitext(self.output)[0]}_violin.png"

        # Chart title, x-axis and y-axis labels
        title = 'Violin Plot of RRCS Scores'
        xlabel = ['Initial Frame', 'Middle Frame', 'Final Frame']
        ylabel = 'RRCS Score'

        # Set the seaborn style
        sns.set_theme(style="whitegrid")

        # Create a figure and axis
        plt.figure(figsize=(12, 8))

        # Create the violin plot
        ax = sns.violinplot(data=data, palette="muted", inner="box")

        # Set the x-tick positions and labels
        ax.set_xticks(range(len(xlabel)))  # Set the positions
        ax.set_xticklabels(xlabel, fontsize=12)

        # Set the title and labels
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel(ylabel, fontsize=14)

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logging.info(f"Violin plot saved to {filename}")

    def reformat_scatter_data(self, rrcs_data):
        """
        Reformat scatter plot data.

        Transforms the original data into (x, y) pairs suitable for plotting a scatter graph.
        The raw data is a dictionary where keys are features and values are lists of multiple tuples,
        each tuple containing a feature value, a metric, and a score.

        Parameters:
        rrcs_data: dict, the original dictionary of data.

        Returns:
        tuple, containing two lists: one for x-axis data and another for y-axis data.
        """
        # Using list comprehension to extract scores from the original data, forming a new list of tuples
        data = [(f, s) for f, scores in rrcs_data.items() for _, _, s in scores if s > 0]
        x, y = zip(*data)

        x = np.array([int(i) for i in x])
        y = np.array([float(i) for i in y])

        return x, y
    
    def reformat_violin_data(self, rrcs_data):
        """
        Reformat violin plot data.
        """
        # Sort the keys of rrcs_data, which are time points, to facilitate the selection of initial, middle, and final time points
        frames = sorted(rrcs_data.keys())
        initial_index = frames[0]
        middle_index = frames[len(frames) // 2]
        final_index = frames[-1]

        initial_frame = [float(s) for _, _, s in rrcs_data[initial_index] if s>0]
        middle_frame = [float(s) for _, _, s in rrcs_data[middle_index] if s>0]
        final_frame = [float(s) for _, _, s in rrcs_data[final_index] if s>0]

        # Combine the data into a list
        return [initial_frame, middle_frame, final_frame]

    def run(self):
        """
        Executes the analysis and outputs the result.

        If the plotting option is configured, it will draw a scatter plot first. 
        Afterwards, regardless of whether plotting is done or not, it will write to the output file.
        """
        # Check if plotting is required
        if self.isplot:
            # Draw a scatter plot
            self.plot_scatter()
            # Draw a violin plot
            self.plot_violin()
        # Write to the output file
        self.write_output()

def main():
    """
    Main function to execute the analysis and plotting.
    """
    # Parse the configuration file and get the basic configuration information
    config_parser = ConfigParser()
    basic_config = config_parser.parse_arguments()

    # Initialize the universe based on the basic configuration, and perform initial checks
    initializer = UniverseInitializer(basic_config)
    initializer.check()

    # Initialize the residue contact analyzer and load the residues for analysis
    analyzer = ResidueContactAnalyzer(initializer.basic)
    analyzer._load_residues()  # Ensureres_list and contact_scores are initialized
    analyzer.analyze_contacts()

    # Initialize the data visualizer and perform the visualization
    DataVisualizer(initializer.basic, analyzer.all_frame_rrcs).run()


if __name__ == "__main__":

    global_start = timeit.default_timer()
    main()
    elapsed = timeit.default_timer() - global_start
    logging.info(
        "The GMX_RRCS program has finished running, with a total elapsed time of "
        + colored(f"{elapsed:.6f} ", 'green')
        + "seconds."
        )

