""" Tests for the Solution class and associated functions in ionmonger """

from ..ionmonger import *
import pytest

@pytest.mark.parametrize("test_file", str(base_model))

def test_Solution(test_file):
    solution = Solution(test_file, 0)

@pytest.mark.parametrize("solution", Solution(test_file))

def test_plot_electronsholes(solution):
    plot_electronsholes(solution)

def test_plot_electricpotential(solution):
    plot_electricpotential(solution)

def test_plot_anionvacancies(solution):
    plot_anionvacancies(solution)

def test_plot_zoomed_anionvacancies(solution):
    plot_zoomed_anionvacancies(solution, E)
    plot_zoomed_anionvacancies(solution, H)

def test_latex_table(solution):
    latex_table(solution)
