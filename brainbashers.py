#! /bin/env python
"""
    Common code for solving the Brainbashers puzzles
"""

from re import search as regex_search
from datetime import datetime, timedelta
import async_timeout


def get_javascript_variable(htmltext, varname):
    """ extract from html text a javascript-based text variable
        of the from varname = 'stringvalue';
        the delimiter can be either single or double quoted
    """
    # pattern = varname + optional whitespaces + '=' + optional whitespaces + 
    #    optional single or double quotes (the quotes are only there if it is a string)
    #    + anything except quotes or ';' 
    #    + optional endquote
    #    + optional whitespace + ';' or '\n'
    pattern = varname+'\s*=\s*["\']*([^"\';]+)["\']*\s*[;\n]'
    result = regex_search(pattern, htmltext)
    if result:
        return result.group(1)
    return None

def daterange(start_date, end_date):
    """ iterator returning a list of consecutive dates """
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(days = n)



class Color:
    # convert 'logical' colors in term colors
    colorscheme = {'white': ('grey','on_white'),
                 'black': ('yellow','on_grey'),
                 'grey': ('blue', 'on_red')
                 }

    invertedcolors = {'white': 'black',
                 'black': 'white',
                 'grey': 'grey',
                 }

    def __init__(self, colorstring):
        assert(colorstring in Color.colorscheme)
        self.color = colorstring


    def invert(self):
        return Color.invertedcolors[self.color]

    def termcolors(self):
        return Color.colorscheme[self.color]


class BrainbasherCell:
    def __init__(self, row, col, value = None, color = Color('grey')):
        self.row, self.col = row, col
        self.value = value
        self.color = color

    def issolved(self):
        raise UnimplementedException


class ContradictionFound(Exception):
    def __init__(self, message = None, error = None):
        super().__init(message)
        self.error =  error


class BrainbasherRaster:
    def _newcell(self, row, col):
        return BrainbasherCell(row,col)

    def __init__(self, nbr_rows, nbr_cols, identity):
        self.nbr_rows, self.nbr_cols = nbr_rows, nbr_cols
        self.raster = [[self._newcell(row,col) for row in range(self.nbr_rows)] for col in range(self.nbr_cols)]
        self.show_solution_method = False # for debugging
        self.identity = identity

    def setcolors(self, colors: list):
        for row in range(self.nbr_rows):
            for col in range(self.nbr_cols):
                self.raster[row][col].color = colors[row * self.nbr_cols + col]

    def setvalues(self, values: list):
        for row in range(self.nbr_rows):
            for col in range(self.nbr_cols):
                self.raster[row][col].value = values[row * self.nbr_cols + col]

    def is_solved(self):
        return not any([self.raster[row][col].is_solved() for row in range(self.nbr_rows) for col in range(self.nbr_cols)])

    def nbr_unsolved_cells(self):
        return len([row for row in range(self.nbr_rows) for col in range(self.nbr_cols) if not self.raster[row][col].is_solved()])

    def make_virtualrow(self, idx, reverse = False):
        assert(-1 < idx < (self.nbr_rows+self.nbr_cols)*2)
        if reverse:
            if idx < self.nbr_rows + self.nbr_cols:
                idx += self.nbr_rows + self.nbr_cols
            else:
                idx -= self.nbr_rows + self.nbr_cols

        steps = [self.nbr_rows, self.nbr_rows + self.nbr_cols, self.nbr_rows * 2 + self.nbr_cols]
        if idx < steps[0]:
            return [self.raster[idx][_] for _ in range(self.nbr_cols)]
        elif idx < steps[1]:
            return [self.raster[_][idx-steps[0]] for _ in range(self.nbr_rows)]
        elif idx < steps[2]:
            return [self.raster[idx-steps[1]][_] for _ in reversed(range(self.nbr_cols))]
        else:
            return [self.raster[_][idx-steps[2]] for _ in reversed(range(self.nbr_rows))]

    def is_border_cell(self, row, col):
        """ True if the cell is at the edge of the board
        """
        return  row == 0 or row == self.nbr_rows - 1 or col == 0 or col == self.nbr_cols - 1

    def cellid(self, virtualrow, idx):
        """ get the raster row and column of a cell by combining the virtualrow and an index
        """
        cell = virtualrow[idx]
        return cell.row, cell.col

    def str_by_row(self, row):
        rowstr = ''
        for col in range(self.nbr_cols):
            rowstr += str(self.raster[row][col])
        return rowstr

    def print_me(self):
        for row in range(self.nbr_rows):
            print(str_by_row(rowstr))



async def solve_brainbasher_task(session, url, html_to_puzzle):
    with async_timeout.timeout(10):
        async with session.request("GET", url = url, allow_redirects = False) as response:
            if response.status == 200:
                print(url)
                html = await response.text()
                if html:
                    puzzle = html_to_puzzle(html)
                    if puzzle:
                        return html_to_puzzle(html).solve()
                    else:
                        return None
            return None

