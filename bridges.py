#! /bin/env python

import pdb
import asyncio
import async_timeout
from datetime import datetime, timedelta
from aiohttp import ClientSession
from termcolor import colored
from collections import defaultdict

from brainbashers import get_javascript_variable, daterange, solve_brainbasher_task, Color, BrainbasherCell, BrainbasherRaster
import brainbashers
from boost import rational_to_count

    
       
class Cell(BrainbasherCell):
    """ a single cell in bridged (including lines and empty space) """

    # convert 'logical' colors in term colors
    colordict = {'white': ('grey','on_white'),
                 'black': ('yellow','on_grey'),
                 'grey': ('red', 'on_grey')
                 }

    def __init__(self, row, col):
        super().__init__(row, col, None, Color('white'))
        """ arrays are arranged such that direction 0 and 2 resp. 1 and 3 are opposite
            so the index of the opposite direction is direction-2
        """
        self.neighbourpos = [None]*4
        self.neighbour = [None]*4
        self.min = [0]*4
        self.max = [2]*4
        self.color = Color('white')
        self.row = row
        self.col = col
        self.island_id = None


    def __str__(self):
        viewstr = ' ' if self.value is None else str(self.value)
        return colored(viewstr, *self.color.termcolors())


    def __repr__(self):
        reprstr = ' ' if self.value is None else str(self.value) + '\n'
        reprstr += str(self.neighbourpos) + '\n'
        reprstr += str(self.min) + '\n'
        reprstr += str(self.max) + '\n'
        reprstr += str(self.row)+ ',' + str(self.col) + '\n'
        reprstr += str(self.island_id) + '\n'
        return reprstr


    def set_color(self, color, check_consistency = True) -> bool:
        """ Change the color of the cell.  """
        self.color = Color(color)


    def verify_solved(self):
        """ Set the cell black if it is solved (min == max) """
        if self.is_solved():
            self.set_color('black')
        for neighbour in [self.neighbour[_] for _ in range(4) if self.neighbour[_]]:
            if neighbour.is_solved:
                neighbour.set_color('black')


    def is_solved(self):
        return self.min == self.max


    def nbr_solved_lines(self):
        return sum(self.min)


    def set_value(self, value):
        self.value = value
        if value is None:
            self.max = [0]*4
            self.set_color('white')
            return
        elif str(value) in '-=|U':
            self.set_color('black')
            return 
        elif value == 1:
            self.max = [1]*4
        self.set_color('grey')


    def set_max_lines(self, neigbour, direction, lines):
        assert(lines >= self.min[direction] and lines >= neigbour.min[direction-2])
        lines = min(self.max[direction], neigbour.max[direction-2], lines)
        self.max[direction] = neigbour.max[direction-2] = lines


    def set_min_lines(self, neigbour, direction, lines):
        assert(lines <= self.max[direction] and lines <= neigbour.max[direction-2])
        lines = max(self.min[direction], neigbour.min[direction-2], lines)
        self.min[direction] = neigbour.min[direction-2] = lines
    

    def nbr_unplaced_lines(self):
        """ How many lines have not yet been placed """
        return self.value - sum(self.min)


    def is_line(self):
        """ Check is the cell is a line segment """
        return str(self.value) in '-=|U'


    def set_neighbour(self, direction, neighbourpos=None, neighbour=None):
        self.neighbourpos[direction] = neighbourpos
        self.neighbour[direction] = neighbour 


class IslandCollection():
    """ Collect information about all island in the raster.
        An island is formed by interconnected cells.
    """
        
    def __init__(self):
        self.islands = dict()


    def create(self, row: int, col: int, cell: Cell) -> int:
        """ Create a new island and return its id """
        island_id = rational_to_count(row, col)
        self.islands[island_id] = [cell]
        return island_id


    def join(self, island1_id, island2_id):
        """ join two islands in a single island, and give all cells the id if the new island """
        if island1_id == island2_id:
            return
        joined, killed = min(island1_id, island2_id), max(island1_id, island2_id)
        for cell in self.islands[killed]:
            cell.island_id = joined
        self.islands[joined].extend(self.islands[killed])
        del self.islands[killed]


    def nbr_unplaced_lines(self, island_id):
        """ How many lines have not yet been placed for this island """
        return sum([_.nbr_unplaced_lines() for _ in self.islands[island_id]])


    def max_lines_within_island(self, island_id, currentlines_between_cells):
        """ check how many lines can at most be placed between two neighbouring (and already connected) cells that are part of the same island
            island_id: the island under investigation
            currentlines_between_cells: the current number of lines connecting the two cells
        """
        # If this island has only two free lines, if you would connect them, it would close the island. Ex. 2--3==2 has 1 line free on 2 and one on 3.
        # 

        island_free = self.nbr_unplaced_lines(island_id)
        return currentlines_between_cells if island_free == 2 else 2


    def max_lines_between_islands(self, island1_id, island2_id):
        """ Determine if the number of connections between two island is limited to zero, one or two
            Connecting two island is limited if placing one or two connections would make the complete island green
        """
        assert(island1_id != island2_id)

        if len(self.islands) <= 2: # if there are only two islands left, they are allowed to be joined to form the solution
            return 2
        
        island1_free = self.nbr_unplaced_lines(island1_id)
        if island1_free > 2: # Even if I connect the islands with 2 lines, there would still be one line free to connect to other islands
            return 2
        
        island2_free = self.nbr_unplaced_lines(island2_id)
        if island2_free > 2: # Even if I connect the islands with 2 lines, there would still be one line free to connect to other islands
            return 2

        if island1_free == island2_free:
            return min(2, island1_free - 1)
        else:
            return 2


class Raster (BrainbasherRaster):
    def _newcell(self, row, col):
        return Cell(row,col)


    def __init__(self, nbr_rows, nbr_cols, identity):
        super().__init__(nbr_rows, nbr_cols, identity)
        self.islands =  IslandCollection()
        self.hints = []


    def setvalues(self, values: list):
        newvalues = [None if _ == '.' else int(_) for _ in values]
        super().setvalues(newvalues)


    def is_valid_position(self, row, col):
        return False if row < 0 or col < 0 or row >= self.nbr_rows or col >= self.nbr_cols else True


    def get_cell_by_pos(self, position):
        """ get a cell at a given position """
        return self.raster[position[0]][position[1]] if position else 0


    def set_island_ids(self, puzzle_string):
        """ Set the unique island ids of all calls with a value
            Do this after all cells got a value and before starting to connect cells with bridges
        """
        for row, col, cell in self.get_all_value_cells():
            cell.island_id = self.islands.create(row, col, cell)


    def set_neighbours(self):
        """ Link each cell to its nearest neighbours

        :return: None
        """

        def set_neighbour_for_direction(cell, direction):
            """ Link each cell to its nearest neighbour in a specified direction
            """
            dxdy = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            dx, dy = dxdy[direction]
            position = (row+dx, col+dy)
            while self.is_valid_position(*position): 
                neighbour = self.get_cell_by_pos(position)
                if neighbour.value is not None: # found a real neighbour
                    if neighbour.is_line(): # a line 
                        cell.set_neighbour(direction) # = None
                    else:   # a value
                        cell.set_neighbour(direction, position, neighbour)
                    return
                else: # an empty field, let the empty field refer to this one and continue looking for a 'real' neighbour
                    neighbour.set_neighbour(direction-2, (row,col), cell)
                position = (position[0]+dx,position[1]+dy)
            # no neighbour found
            cell.set_neighbour(direction) # = None
            

        for row, col, cell in self.get_unsolved_cells():
            if cell.value:
                for direction in range(4):
                    set_neighbour_for_direction(cell, direction)


    def get_all_value_cells(self):
        for row in range(self.nbr_rows):
            for col in range(self.nbr_cols):
                cell = self.raster[row][col]
                if isinstance(cell.value, int):
                    yield row, col, self.raster[row][col]


    def get_unsolved_cells(self):
        """ return any cell from the raster that is still unsolved
        """
        for row, col, cell in self.get_all_value_cells():
            if not cell.is_solved():
                yield row, col, self.raster[row][col]


    def nbr_unsolved_cells(self):
        return len([_ for _ in self.get_unsolved_cells()])


    def draw_line(self, count, beginpos, endpos):
        """ draw count lines between two cells
            assume cells are on a horizontal or vertical line
        """
        # don't get confused in the raster, dx is horizontal displacement, y is vertical 
        dy = 0 if endpos[0] == beginpos[0] else 1 if endpos[0] > beginpos[0] else -1
        dx = 0 if endpos[1] == beginpos[1] else 1 if endpos[1] > beginpos[1] else -1
        char = '|U'[count-1] if dx == 0 else '-='[count-1]
        pos = (beginpos[0]+dy, beginpos[1]+dx)
        if self.get_cell_by_pos(pos).value == char:
            # a line is already present, so we can stop
            return

        while pos != endpos:
            linecell = self.get_cell_by_pos(pos)
            linecell.set_value(char)
            orthogonal_directions = (1,3) if dx == 0 else (0,2)
            for direction in orthogonal_directions:
                neighbour = linecell.neighbour[direction]
                if neighbour:
                    neighbour.neighbourpos[direction-2] = None
                    neighbour.neighbour[direction-2] = None
            pos = (pos[0]+dy, pos[1]+dx)

        begincell = self.get_cell_by_pos(beginpos)
        endcell = self.get_cell_by_pos(endpos)
        self.islands.join(begincell.island_id, endcell.island_id)
 
    def solve(self):
        # continue looking for new solution parts as long as our strategies result in more new lines being found
        previous_solved_lines = None
        solved_lines = 0
        while solved_lines != previous_solved_lines:
            previous_solved_lines = solved_lines
            for row, col, cell in self.get_unsolved_cells():
                # update min/max values based on data from neighbours
                for direction in [ _ for _ in range(4) if cell.max[_] > cell.min[_]]:
                    neighbour = cell.neighbour[direction]
                    if not neighbour:
                        cell.min[direction] = 0
                        cell.max[direction] = 0
                    else:
                        if neighbour.value == 1:
                            cell.set_max_lines(neighbour, direction, 1 if cell.value > 1 else 0)
                        elif neighbour.value == 2:
                            cell.set_max_lines(neighbour, direction, 2 if cell.value > 2 else 1)
                        cell.set_max_lines(neighbour, direction, 2 if cell.value > 1 else 1)
                        cell.set_min_lines(neighbour, direction, 0)

                for direction in [ _ for _ in range(4) if cell.max[_] > cell.min[_]]:
                    # update min/max values based on data from other directions of the own cell
                    maxsum_other_directions = sum(cell.max[_] for _ in range(4) if _ != direction)
                    minsum_other_directions = sum(cell.min[_] for _ in range(4) if _ != direction)
                    cell.set_min_lines(cell.neighbour[direction], direction, cell.value - maxsum_other_directions)
                    cell.set_max_lines(cell.neighbour[direction], direction, cell.value - minsum_other_directions)
                     
                    cell.verify_solved() # color the cell if it is solved now
                    # draw known lines
                    if cell.min[direction] > 0:
                        self.draw_line(cell.min[direction], (row,col), cell.neighbourpos[direction])
                
                cell.verify_solved() # color the cell if it is solved now

            for row, col, cell in self.get_unsolved_cells():
                # update min/max values based on the rule that no islands should be formed
                for direction in [ _ for _ in range(4) if cell.max[_] > cell.min[_]]:
                    neighbour = self.get_cell_by_pos(cell.neighbourpos[direction])
                    if neighbour:
                        if cell.island_id == neighbour.island_id:
                            max_connecting_lines = self.islands.max_lines_within_island(cell.island_id, cell.min[direction])
                        else:
                            max_connecting_lines = self.islands.max_lines_between_islands(cell.island_id, neighbour.island_id)
                        cell.set_max_lines(neighbour, direction, max_connecting_lines)
                            
                cell.verify_solved() # color the cell if it is solved now

            solved_lines = sum([ cell.nbr_solved_lines() for (_,_,cell) in self.get_all_value_cells() ])

        unsolved = len([_ for _ in self.get_unsolved_cells()])
        if unsolved == 0:
            return self
        else:
            print('could not solve this:')
            self.print_me()
            return self

    def print_me(self):
        for row in range(self.nbr_rows):
            rowstr = ''
            for col in range(self.nbr_cols):
                rowstr = rowstr + str(self.raster[row][col])
            print(rowstr)


    def print_islands(self):
        """ for debugging, show the islands """
        for row in range(self.nbr_rows):
            rowstr = ''
            for col in range(self.nbr_cols):
                if self.raster[row][col].island_id:
                    rowstr = rowstr + str(self.raster[row][col].island_id)
                else:
                    rowstr = rowstr + ' '
            print(rowstr)

        print(self.islands.islands.keys())



def puzzle_from_html(html) -> Raster:
    puzzle_string = get_javascript_variable(html, 'lcpuzzle')
    if not puzzle_string:   # received html was not a valid puzzle
        return None
    puzzle_size = int(get_javascript_variable(html, 'lnsize'))
    puzzle_id = get_javascript_variable(html, 'lcpuzzletext')
    puzzle = Raster(puzzle_size, puzzle_size, puzzle_id)
    puzzle.setvalues(puzzle_string)
    puzzle.set_island_ids(puzzle_string)
    puzzle.set_neighbours()
    return puzzle


def puzzle_urls(whichs=[7,10,15,20]):
    """ generate urls for all possible bridges puzzles
    """
    tomorrow = datetime.today()+timedelta(days=1)
    for urldate in daterange(datetime(datetime.today().year,7,26), tomorrow):
        for diff in ["Easy","Medium","Hard"]:
            for which in whichs:
                yield f'https://brainbashers.com/showbridges.asp?date={urldate.month:02}{urldate.day:02}&diff={diff}&size={which}'


async def main(loop):
    async with ClientSession(loop=loop) as session:
        for url in puzzle_urls():
            print('downloaded ', url)
            solution = await solve_brainbasher_task(session, url, puzzle_from_html)
            if solution and not solution.is_solved():
                print(solution.identity)
                solution.print_me()


def testmain():
    """ main procedure for debugging a single puzzle. """
    import requests
    url = 'https://brainbashers.com/showbridges.asp?date=0726&diff=Easy&size=15'
    response = requests.get(url)
    puzzle = puzzle_from_html(response.text)
    puzzle.solve()
    puzzle.print_me()


if __name__ == '__main__':
    # uncomment the next two lines for debugging
    #testmain()
    #quit()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))
