#! /bin/env python

import pdb
import asyncio
import async_timeout
from datetime import datetime, timedelta
from aiohttp import ClientSession
from termcolor import colored
from collections import defaultdict
from re import search as regex_search

"""TODO
suppose at a border, we have
7
75
55

if corner 5 = white, then middle 7 is cornered => 5 is black
"""

class Cell:
    """ a single cell in a Hitori raster """

    # convert 'logical' colors in term colors
    colordict = {'white': ('grey','on_white'),
                 'black': ('yellow','on_grey'),
                 'grey': ('blue', 'on_red')
                 }

    def __init__(self):
        self.value = None
        self.color = "grey"

    def __str__(self):
        return colored(str(self.value), *self.colordict[self.color])

    def set_color(self, color, check_consistency = True) -> bool:
jan@Jupiter ~/programming/DailyHitori $ cat daily_hitory.py
#! /bin/env python

import pdb
import asyncio
import async_timeout
from datetime import datetime, timedelta
from aiohttp import ClientSession
from termcolor import colored
from collections import defaultdict
from re import search as regex_search

"""TODO
suppose at a border, we have
7
75
55

if corner 5 = white, then middle 7 is cornered => 5 is black
"""

class Cell:
    """ a single cell in a Hitori raster """

    # convert 'logical' colors in term colors
    colordict = {'white': ('grey','on_white'),
                 'black': ('yellow','on_grey'),
                 'grey': ('blue', 'on_red')
                 }

    def __init__(self):
        self.value = None
        self.color = "grey"

    def __str__(self):
        return colored(str(self.value), *self.colordict[self.color])

    def set_color(self, color, check_consistency = True) -> bool:
        """ Change the color of the cell.
            If check_consistency == True, verify that only valid transistions are made.
                Raster.ContradictionFound is thrown upon invalid colorchanges.
                a grey cell can get any color
                a colored cell may not be overwritten by another color.
                overwriting a colored cell by the same color returns False, otherwise True
        """
        if self.color == "grey" or not check_consistency:
            self.color = color
            return True
        elif self.color == color:
            return False
        else:   # a contradiction happens: trying to turn a white cell black or vice versa
            raise Raster.ContradictionFound()



class Raster:

    class ContradictionFound(Exception):
        def __init__(self, message = None, error = None):
            super().__init__(message)
            self.error = error

    def __init__(self, nbr_rows, nbr_cols, identity):
        self.raster = [[Cell() for row in range(nbr_rows)] for col in range(nbr_cols)]
        self.nbr_rows, self.nbr_cols = nbr_rows, nbr_cols
        self.show_solution_method = False # for debugging
        self.identity = identity

    def setvalues(self, values: list):
        for row in range(self.nbr_rows):
            for col in range(self.nbr_cols):
                self.raster[row][col].value = values[row * self.nbr_cols + col]

    def nbr_unsolved_cells(self):
        return sum([self.raster[row][col].color == "grey" for row in range(self.nbr_rows) for col in range(self.nbr_cols) if self.raster[row][col].color == "grey"])

    def is_solved(self):
        return not any([self.raster[row][col].color == "grey" for row in range(self.nbr_rows) for col in range(self.nbr_cols)])

    def set_color(self, row, col, color, guessresults = []):
            if self.raster[row][col].set_color(color):
                guessresults.append((row,col))
                return True
            return False

    def make_black(self, row, col, guessresults = []):
        if guessresults and self.is_closing_cell(row, col):
            raise Raster.ContradictionFound

        if self.set_color(row, col, "black", guessresults):
            if self.show_solution_method:
                print(f'row {row+1}, col {col+1}: black')
            neighbour_cols = [_ for _ in [col - 1, col + 1] if _ in range(self.nbr_cols)]
            for other_col in neighbour_cols:
                self.make_white(row, other_col, guessresults)

            neighbour_rows = [_ for _ in [row - 1, row + 1] if _ in range(self.nbr_rows)]
            for other_row in neighbour_rows:
                self.make_white(other_row, col, guessresults)

    def make_white(self, row, col, guessresults):
        if self.set_color(row, col, "white", guessresults):
            if self.show_solution_method:
                print(f'row {row+1}, col {col+1}: white')

            for other_col in [_ for _ in range(self.nbr_cols) if _ != col]:
                if self.raster[row][other_col].value == self.raster[row][col].value:
                    self.make_black(row, other_col, guessresults)

            for other_row in [_ for _ in range(self.nbr_rows) if _ != row]:
                if self.raster[other_row][col].value == self.raster[row][col].value:
                    self.make_black(other_row, col, guessresults)

    def check_corner_pairs(self):
        """
            If two cells almost in a corner form a pair, one cell can be known:
            E.g cells (0,1)and (1,1) have the same value, so one must be white, and the other black
            if (0,1) is black, then (1,0) must be whit to prevent isolating (0,0)
            if (1,1) is black, the (1,0) must be a white neighbour
            => (1,0) is white
            This procedure needs to be called only once, as it does not depend on what you find later.
            That is why it doesn't backtrack and thus cannot be used in guess()
        """
        corners = [(row,col) for row in (1,self.nbr_rows-2) for col in (1,self.nbr_cols-2)]
        for (corner_row,corner_col) in corners:
            row_delta = 1 if corner_row > 1 else -1
            col_delta = 1 if corner_col > 1 else -1
            if self.raster[corner_row][corner_col].value == self.raster[corner_row+row_delta][corner_col].value:
                self.make_white(corner_row, corner_col + col_delta, guessresults = [])
            if self.raster[corner_row][corner_col].value == self.raster[corner_row][corner_col+col_delta].value:
                self.make_white(corner_row+row_delta, corner_col, guessresults = [])


    def make_centered_white(self):
        """ if a cell lies between two cells with the same value, then that cell cannot be black because that would make these two values both white
                Therefore, the cell must be white
            This procedure needs to be called only once, as it does not depend on what you find later.
            That is why it doesn't backtrack and thus cannot be used in guess()
        """
        for row in range(self.nbr_rows):
            for col in range(1, self.nbr_cols - 1):
                if self.raster[row][col - 1].value == self.raster[row][col + 1].value:
                    self.make_white(row, col, guessresults = [])

        for col in range(self.nbr_cols):
            for row in range(1, self.nbr_rows - 1):
                if self.raster[row - 1][col].value == self.raster[row + 1][col].value:
                    self.make_white(row, col, guessresults = [])

    def is_border_cell(self, row, col):
        return  row == 0 or row == self.nbr_rows - 1 or col == 0 or col == self.nbr_cols - 1

    def diagonal_neighbours(self, row, col):
        """ returning the diagonal neighbours of the cell at location row, col
        """
        neighbour_rows = [_ for _ in [row - 1, row + 1] if _ in range(self.nbr_cols)]
        neighbour_cols = [_ for _ in [col - 1, col + 1] if _ in range(self.nbr_cols)]
        return  [(x, y) for x in neighbour_rows for y in neighbour_cols]

    def is_path_to_border_or_loop(self, row, col, from_cell, origincell):
        """ return 1 if it is possible to reach a border, coming from
                from_cell (a tuple with (row, col) of that cell) through the cell defined by row, col
        """
        if (row, col) == origincell:
            return 2

        if self.raster[row][col].color != "black":
            return 0

        if self.is_border_cell(row, col):
            return 1

        nbr_reachable_borders = 0
        for other_row, other_col in self.diagonal_neighbours(row, col):
            if (other_row, other_col) != from_cell:
                if self.is_path_to_border_or_loop(other_row, other_col, from_cell=(row, col), origincell=origincell):
                    nbr_reachable_borders += 1
        return nbr_reachable_borders

    def is_closing_cell(self, row, col):
        """ return true if making this cell black would result in a closed region
        """
        nbr_reachable_borders = 1 if self.is_border_cell(row, col) else 0

        for other_row, other_col in self.diagonal_neighbours(row, col):
            nbr_reachable_borders += self.is_path_to_border_or_loop(other_row, other_col, from_cell=(row, col), origincell=(row,col))
            if nbr_reachable_borders > 1:
                return True
        return False


    def set_closing_cells_white(self, guessresults):
        """ set any cell that closes off a region white (and calculate the consequences)
        """
        """ TODO: make get_any_grey_cell an iterator and loop over that
        """
        for row, col in self.get_any_grey_cell():
            if self.is_closing_cell(row, col):
                self.make_white(row, col, guessresults)

    def get_any_grey_cell(self):
        """ return any cell from the raster that is still grey
        """
        for row in range(self.nbr_rows):
            for col in range(self.nbr_cols):
                if self.raster[row][col].color == "grey":
                    yield row,col

    def set_closing_cells_white_until_solved(self, guessresults):
        unsolved = self.nbr_unsolved_cells()
        previous_unsolved = None
        while unsolved != previous_unsolved:
            previous_unsolved = unsolved
            self.set_closing_cells_white(guessresults)
            unsolved = self.nbr_unsolved_cells()
        return unsolved == 0


    def undo_guess(self, guessresults):
        for row, col in guessresults:
            self.raster[row][col].set_color("grey", check_consistency = False)


    def guess(self, guessresults):
            row, col = next(self.get_any_grey_cell())
            try:
                white_guessresults = []
                print('guess: ', row, ', ', col, '=white ?')
                self.make_white(row, col, white_guessresults)
                if not self.set_closing_cells_white_until_solved(white_guessresults):
                    self.guess(white_guessresults)    # either this solves it, or it throws
            except Raster.ContradictionFound:
                self.undo_guess(white_guessresults)
                # our guess (white) was not correct, so we assume black
                # if this also throws, it means that something was wrong before we were called,
                # so we don't catch that exception but let the exception bubble up
                print('guess: ', row, ', ', col, '=black')
                self.make_black(row, col, guessresults)
                if not self.set_closing_cells_white_until_solved(guessresults):
                    self.guess(guessresults)


    def exclude_for_pair(self, cellvalues):
        """ look for two elements with the same value that are neighbours
            all other similar values on that line must be black
        """
        locations=defaultdict(lambda: [])
        for idx in range(len(cellvalues)):
            locations[cellvalues[idx]].append(idx)
        must_be_black = []
        for value in locations:
            for idx in range(len(locations[value]) - 1):
                if locations[value][idx] + 1  == locations[value][idx + 1]:
                    must_be_black += locations[value][0:idx] + locations[value][idx+2:]
                    break
        return must_be_black


    def check_pairs(self):
        for row in range(self.nbr_rows):
            for col in self.exclude_for_pair([self.raster[row][col].value for col in range(self.nbr_cols)]):
                    self.make_black(row,col)
        for col in range(self.nbr_cols):
            for row in self.exclude_for_pair([self.raster[row][col].value for row in range(self.nbr_rows)]):
                self.make_black(row,col)


    def solve(self):
        self.make_centered_white()
        self.check_corner_pairs()
        self.check_pairs()
        if not self.set_closing_cells_white_until_solved(guessresults = []):
            print('could not solve this:')
            self.print_me()
            self.guess(guessresults = [])
            self.print_me()
        return self


    def print_me(self):
        for row in range(self.nbr_rows):
            rowstr = ''
            for col in range(self.nbr_cols):
                rowstr += str(self.raster[row][col])
            print(rowstr)


def getvariable(htmltext, varname):
        pattern = varname+'\s*=\s*[\']*([^"\';]+)[\']*;'
        result = regex_search(pattern, htmltext)
        if result:
            return result.group(1)
        return None


def puzzle_from_html(html) -> Raster:
    puzzle_string = getvariable(html, 'lcpuzzle')
    size = getvariable(html, 'lnsize')
    puzzle_size = int(getvariable(html, 'lnsize'))
    puzzle_id = getvariable(html, 'lcpuzzletext')
    puzzle = Raster(puzzle_size, puzzle_size, puzzle_id)
    puzzle.setvalues([int(_, 16) for _ in puzzle_string])
    return puzzle


async def solve_brainbasher_task(session, url):
    with async_timeout.timeout(10):
        async with session.request("GET", url = url, allow_redirects = False) as response:
            if response.status == 200:
                print(url)
                html = await response.text()
                if html:
                    return puzzle_from_html(html).solve()
            #print(f'No html for {url}')
            return None


def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(days = n)


def hitory_urls(sizes=[5,6,7,8,9], levels=[1,2], include_weekly = True):
    """ generate urls for all possible hitory puzzles
        weekly puzzles (size 12) only works for the last week
    """
    tomorrow = datetime.today()+timedelta(days=1)
    for urldate in daterange(datetime(datetime.today().year,6,10), tomorrow):
        for level in levels:
            for size in sizes:
                yield f'https://brainbashers.com/showhitori.asp?date={urldate.month:02}{urldate.day:02}&size={size}&diff={level}'
    for urldate in daterange(datetime.today()-timedelta(days=7), tomorrow):
        level, size = 1, 12
        yield f'https://brainbashers.com/showhitori.asp?date={urldate.month:02}{urldate.day:02}&size={size}&diff={level}'



async def main(loop):
    async with ClientSession(loop=loop) as session:
        for url in hitory_urls():
            solution = await solve_brainbasher_task(session, url)
            if solution and not solution.is_solved():
                print(solution.identity)
                solution.print_me()

def testmain():
        import requests
        url = 'https://brainbashers.com/showhitori.asp?date=0617&size=7&diff=2'
        response = requests.get(url)
        puzzle = puzzle_from_html(response.text)
        puzzle.solve()
        puzzle.print_me()

if __name__ == '__main__':
    #testmain()
    #quit()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))
