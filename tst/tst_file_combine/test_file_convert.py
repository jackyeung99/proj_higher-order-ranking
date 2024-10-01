import sys
import os
import pytest

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from datasets.utils.convert_raw_files import *
from src import *

class TestSocCombination:

    @classmethod
    def setup_class(cls):
        cls.data_dir = os.path.join(repo_root, 'tst', 'tst_file_combine', 'tst_files')
        cls.processed_files = os.path.join(cls.data_dir, 'processed_tst_files')
        cls.test_edges = os.path.join(cls.data_dir, '99999_edges.txt')
        cls.test_nodes = os.path.join(cls.data_dir,'99999_nodes.txt')

    def test_read(self):
        file = os.path.join(self.data_dir, '99999_edges.txt')
        data, pi_values = read_edge_list(file)

        assert len(data) == 3
        assert len(pi_values) == 5
        
    def test_name_conversion(self):
        id_to_name = get_alternative_names(self.test_nodes)
      
        assert len(id_to_name) == 5
        assert id_to_name[1] == 'a'
        assert id_to_name[2] == 'b'

    def test_game_conversion(self):
        id_to_name = get_alternative_names(self.test_nodes)
        data, pi_values = read_edge_list(self.test_edges)
        named_games = convert_id_to_name(data, id_to_name)

        assert len(named_games) == len(data)
        assert named_games[('d','a','b')] == 2

    def test_grouping(self):
        file_directory = os.path.join(self.data_dir, 'soc_tst_files')
        grouping = group_soi(file_directory)

        print(grouping)
        assert len(grouping) == 3
        assert all(len(x) for x in grouping.values())

    def test_combine_soi(self):

        file_directory = os.path.join(self.data_dir, 'soc_tst_files')
        grouping = group_soi(file_directory)
        combine_soi(grouping, file_directory, self.processed_files)
        assert '00001_nodes.txt' in os.listdir(self.processed_files)
        assert '00001_edges.txt' in os.listdir(self.processed_files)

        data, pi_values = read_edge_list(os.path.join(self.processed_files, '10000_edges.txt'))

        assert len(pi_values) == 5
        assert len(data) == 3   
        assert sum([v for k,v in data.items()]) == 5

    def test_combined_files(self):
        data, pi_values = read_edge_list(os.path.join(self.processed_files, '00042_edges.txt'))
        assert len(pi_values) == 24


    def test_mapping(self):
        data, pi_values = read_edge_list(os.path.join(self.processed_files, '10000_edges.txt'))

        assert (3, 1, 4) in data 
        assert data[(3, 1, 4)] == 2


