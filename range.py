#! /bin/env python

import pdb
import datetime
import time
import itertools
from enum import IntEnum
from collections import deque
from termcolor import colored

class slopes(IntEnum):
    horizontal = 0
    vertical = 1

    def inverse(self):
        ''' turn horizontal into vertical and vice versa '''
        return slopes.horizontal if self == slopes.vertical else slopes.vertical


directions=[-1,1]

headings=list(itertools.product(slopes, directions))
def inverse_heading(heading):
    return (heading[0], -heading[1])


class base_cell():

    def __init__(self, cellid):
        self.cellid = cellid
        self.solved = 0 # 0: cell is completely unsolved, 1, we know minmax, but the cell may not have been surrounded, 2 the cell is fully solved
        self.color = 'G' if cellid else 'D'
        self.value = None


    def is_solved(self, solution_states = [2]):
        return self.solved in solution_states


    def set_solved(self, value = 2):
        self.solved = value


    def set_value(self, value):
        self.make_white()
        self.value = value


    def set_color(self, color):
        if self.color not in [color, 'D']:
            self.color = color
            return True
        return False


    def make_white(self):
        if self.color == 'B':
            self.show()
        assert(self.color != 'B')
        return self.set_color('W')


    def make_black(self):
        return self.set_color('B')


    def is_value(self):
        return self.value is not None


    def is_black(self):
        return self.color in 'DB'


    def is_grey(self):
        return self.color == 'G'


    def is_white(self):
        return self.color == 'W'


    def show(self):
        ''' show the content of a cell (for debugging) '''
        print(self.cellid)
        print(self.color)
        print('value: ', self.value)
        print('solved: ', self.solved)


    def __str__(self):
        ''' return a colorized string representing the cell when printed on terminal.
            to be used for printing the raster
        '''
        if self.color is 'G':
            return colored('__', 'white', 'on_grey')
        elif self.color == 'B':
            return colored('__', 'red', 'on_red')
        elif self.color == 'W':
            if self.value:
                return colored(f'{self.value:2}', 'grey', 'on_green', attrs=['dark'])
            else:
                return colored('__', 'green', 'on_green')
        else:
            return 'DD'


class rastercell(base_cell):
    ''' a cell that is aware of its position in the raster '''

    def __init__(self, cellid, raster = None):
        super().__init__(cellid)
        self.raster = raster
        if cellid:
            row, col = cellid
            self.minmax = { key : { 'min': 1, 'max' : value } for key,value in zip(slopes, [raster.width, raster.height])}


    def set_value(self, value):
        super().set_value(value)
        for slope in slopes:
            self.minmax[slope]['max'] = min(self.value, self.minmax[slope]['max'])


    def make_white(self):
        result = super().make_white()
        if result:
            self.raster.nbr_unsolved_cells -= 1
        return result


    def make_black(self):
        result = super().make_black()
        if result:
            self.raster.nbr_unsolved_cells -= 1
        return result


    def jump_to_neighbour(self, heading, steps = 1):
        ''' get the neighbour that is 'steps' steps removed from this cell following the given heading '''
        slope, direction = heading
        row, col = self.cellid
        if slope == slopes.horizontal:
            col += direction * steps
        else:
            row += direction * steps
        try:
            return self.raster.getcell(row, col)
        except IndexError:
            return rastercell(None) # return a bordercell (always behaves like a black one)


    def is_bordercell(self):
        ''' True if the cell is at the border of the raster '''
        return self.cellid[0] in [0, self.raster.height-1] or self.cellid[1] in [0, self.raster.width-1]


    def diagonal_neighbours(self):
        ''' yield all diagonal neighbours '''
        row, col = self.cellid
        for rowadd, coladd in itertools.product(directions, directions):
            try:
                yield self.raster.getcell(row+rowadd, col+coladd)
            except IndexError:
                pass


    def show(self):
        ''' show the content of a cell (for debugging) '''
        super().show()
        print(self.minmax)


class extent:

    def __init__(self, cell, slope, conditionfunction):
        headings_for_slope = [ _ for _ in headings if _[0] == slope]
        ends = [ self.get_extent_end_by_heading(cell, heading, conditionfunction) for heading in headings_for_slope ]
        self.slope = slope
        self.extent = self.get_rectangle(*ends)


    def set_min_for_extent(self):
        ''' find the valuecell with the smallest minvalue and set all other cells in the extent to this minvalue '''
        valuecells = [ _ for _ in self.extent if _.is_value()]
        min_value = max(*[_.minmax[slope]['min'] for _ in valuecells],
                        *[_.value - _.minmax[slope.inverse()]['max'] + 1 for _ in valuecells],
                        len(self.extent)
                       )
        for cell in valuecell:
            cell.minmax[slope]['min'] = min_value



class rangeraster:

    def __init__(self, height: int, width: int, name=None):
        self.height = height
        self.width = width
        self.name = name
        self.raster = [[rastercell((row,col), self)  for col in range(width)] for row in range(height)]
        self.updated = True
        self.nbr_unsolved_cells = height*width


    def getcell(self, row, col):
        ''' return the cell at position row, col '''
        if row < 0 or col < 0:
            raise IndexError
        return self.raster[row][col]


    def get_distance(self, from_cell, to_cell):
        ''' return the distance (only through horizontal or vertical moves) from one cell to the other '''
        return sum(to_cell.cellid) -sum(from_cell.cellid)


    def allcells(self, solution_states = [0,1,2]):
        ''' a generator yielding all cells in the raster that have a solution state in solution_states '''
        for row, col in itertools.product(range(self.height), range(self.width)):
            cell = self.raster[row][col]
            if cell.solved in solution_states:
                yield cell


    def getallconditioncells(self, condition_function, solution_states):
        ''' a generator yielding all cells obeying a certain condition '''
        for cell in self.allcells(solution_states):
            if condition_function(cell):
                yield cell


    def getallvaluecells(self, solution_states = [0,1,2]):
        ''' a generator yielding all cells having a value in the raster '''
        for cell in self.getallconditioncells(rastercell.is_value, solution_states):
                yield cell


    def getallgreycells(self, solution_states = [0,1,2]):
        ''' a generator yielding all cells having a value in the raster '''
        for cell in self.getallconditioncells(rastercell.is_grey, solution_states):
                yield cell


    def get_rectangle(self, cell1, cell2):
        ''' get a list with all cells in the rectangle with as corners the cells at row1, col1 and row2, col2
            this will mostly be used to get a horizontal or vertical extent
        '''
        row1, col1 = cell1.cellid
        row2, col2 = cell2.cellid
        rowstep = 1 if row2 > row1 else -1
        colstep = 1 if col2 > col1 else -1
        return [ self.raster[row][col] for row, col in itertools.product(range(row1, row2 + rowstep, rowstep), range(col1, col2 + colstep, colstep)) ]


    def get_extent_end_by_heading(self, cell, heading, conditionfunction):
        ''' get the last cell you can reach in one heading passing only through cells that comply to a specific condition '''
        neighbour = cell
        while conditionfunction(neighbour):
            lastneighbour = neighbour
            neighbour = neighbour.jump_to_neighbour(heading)
        return lastneighbour


    def get_extent(self, cell, slope, conditionfunction):
        headings_for_slope = [ _ for _ in headings if _[0] == slope]
        ends = { heading: self.get_extent_end_by_heading(cell, heading, conditionfunction) for heading in headings_for_slope }
        return self.get_rectangle(*ends.values()), ends


    def get_extent_by_color(self, cell, slope):
        ''' get a list of all cells that are connected to the given cell for the given slope with a single color '''
        return self.get_extent(cell, slope, lambda _: _.color == cell.color)


    def get_greywhite_extent(self, cell, slope):
        ''' get a list of all grey or white cells that are connected to the given cell for for the given slope '''
        return self.get_extent(cell, slope, lambda _: not _.is_black())


    def is_solved(self):
        return self.nbr_unsolved_cells <= 0


    def get_last_greywhite_cell(self, cell, heading, nextcell = False):
        ''' For the given cell, get the nearest valid cell bordering the border or a black cell
            unless nextcell = True, then you want one cell beyond that
        '''
        neighbour = cell
        while not neighbour.is_black():
            lastneighbour = neighbour
            neighbour = neighbour.jump_to_neighbour(heading)
        return neighbour if nextcell else lastneighbour


    def get_last_cell_by_color(self, cell, heading, nextcell = False):
        ''' For the given cell, get the last cell of the same color reachable in the given heading.
            unless nextcell = True, then you want one cell beyond that
        '''
        neighbour = cell
        while neighbour.color == cell.color:
            lastneighbour = neighbour
            neighbour = neighbour.jump_to_neighbour(heading)
        return neighbour if nextcell else lastneighbour


    def find_max_value(self, cell, slope):
        ''' return a max value for a value cell that is not too large.

            we do this by first assuming that one side of the extent that the value cell is in is fixed.
            then we check how far we can extend and not get an invalid situation. We remember the last valid cell in that direction.
            Then we do the same in the other direction. the maximum length we can use without violating any condition
            would be the distance between the two 'last valid cells'

            When going in a direction, we must stop if:
            * we reach a black cell. The last valid cell is the last cell of the previous extent;
            * we reach a grey extent:
                * adding the full extent would be longer than the max length: we can fit the max length in only one directtion, so we can return the original max length.
                * adding the full extent would be exactly the max length. If the next extent is white, then the length would be too big, so the last valid cell would be the last cell of the grey extent minus one.
            * we reach a white extent:
                * the white extent contains value cells with a max value lower than the max length:
                    for this iteration, we will either not use the extent, so the last valid cell is the cell before the last one in the previous extent,
                    or it will be included and then max_length will not be greater than this.
                * otherwise we can treat it exactly like a grey extent, with the exception that we MUST reach the end of the extent
        '''
        assert(cell.is_value())
        max_extent_len = min(cell.minmax[slope]['max'],
                             cell.value - cell.minmax[slope.inverse()]['min'] + 1
                            )
        new_max_extent_len = dict()
        ends = dict()
        for direction in directions:
            heading = (slope, direction)
            first_cell_of_next_extent = cell
            extent_len_so_far = 0
            while True:
                if first_cell_of_next_extent.is_black():
                    last_valid_cell = last_cell_of_current_extent
                    new_max_extent_len[direction] = extent_len_so_far
                    self.fit_min_cell(cell, heading, last_valid_cell)
                    break;

                else:   # white or grey
                    current_extent, last_cells = self.get_extent_by_color(first_cell_of_next_extent, slope)
                    last_cell_of_current_extent = last_cells[heading]
                    pre_last_cell_of_current_extent = last_cell_of_current_extent.jump_to_neighbour(inverse_heading(heading))
                    first_cell_of_next_extent = last_cell_of_current_extent.jump_to_neighbour(heading)
                    extent_len_so_far += len(current_extent)

                    if last_cell_of_current_extent.is_grey():
                        if extent_len_so_far > max_extent_len:
                            # we were able to fit the max length in this direction only
                            return max_extent_len

                        else:
                            last_valid_cell = pre_last_cell_of_current_extent
                            new_max_extent_len[direction] = extent_len_so_far - 1

                    else:   # white cell
                        current_extent_value_cells = [_.minmax[slope]['max'] for _ in current_extent if _.is_value()]
                        if current_extent_value_cells and max(current_extent_value_cells) < max_extent_len:
                                # this is the trickiest case.
                                #   we could lower the max value and continue, but then we may need to backtrack.
                                #   instead we know that the maxlen in this direction is determined by either stopping data from the previous extent
                                #   or by lowering the max value and continuing. However, the continuation will be done by another iteration
                                # last valid cell is the one from the previous grey extent
                                new_max_extent_len[direction] = max(new_max_extent_len[direction], max(current_extent_value_cells))
                                break

                        else:
                            if extent_len_so_far > max_extent_len:
                                # using the current extent bring us too far. we use the values of the previous grey extent
                                self.fit_min_cell(cell, heading, last_valid_cell)
                                break

                            else:
                                last_valid_cell = last_cell_of_current_extent
                                new_max_extent_len[direction] = extent_len_so_far

            ends[direction] = last_valid_cell

        fit_distance = self.get_distance(*ends.values()) + 1
        fit_distance = max(fit_distance, *new_max_extent_len.values())
        return fit_distance


    def find_max_values(self):
        for cell in self.getallvaluecells(solution_states = [0]):
            for slope in slopes:
                cell.minmax[slope]['max'] = min(cell.minmax[slope]['max'], self.find_max_value(cell, slope))

        self.update_minmax()


    def print_minmax(self):
        ''' print minmax values for all value cells (for debugging) '''
        for cell in self.getallvaluecells():
            print(cell.cellid, cell.value, cell.minmax)
            #assert(cell.value + 1 == cell.minmax[slopes.horizontal]['max'] + cell.minmax[slopes.vertical]['min'])
            #assert(cell.value + 1 == cell.minmax[slopes.horizontal]['min'] + cell.minmax[slopes.vertical]['max'])


    def update_minmax(self, cells = None):
        ''' calculate the minmax values based on the minmax values of the other slope, such that the condition
            min for one slope + max for other slope == cellvalue + 1
        '''
        loopchange = True
        while loopchange:
            loopchange = False
            for cell in self.getallvaluecells(solution_states = [0,1]):
                    extent = { slope: self.get_extent_by_color(cell, slope)[0] for slope in slopes }
                    greywhite_extent = { slope: self.get_greywhite_extent(cell, slope)[0] for slope in slopes }
                    if sum([len(extent[slope]) for slope in slopes]) == cell.value + 1:
                        cell.set_solved()
                        self.surround_solved_cell(cell)
                    for slope in slopes:
                        valuecells_in_extent = [ _ for _ in extent[slope] if _.is_value()]
                        other_slope = slope.inverse()
                        if cell.solved:
                            # propagate the solution to all other cells in the extent
                            for valuecell in valuecells_in_extent:
                                valuecell.minmax[slope]['min'] = valuecell.minmax[slope]['max'] = len(extent[slope])
                                valuecell.minmax[other_slope]['min'] = valuecell.minmax[other_slope]['max'] = valuecell.value + 1 - len(extent[slope])
                                self.fit_min_for_known_length(valuecell, other_slope)
                        else:
                            pre = cell.minmax[slope].copy()
                            cell.minmax[slope]['min'] = max(*[mycell.minmax[slope]['min'] for mycell in valuecells_in_extent],
                                                              cell.value - cell.minmax[other_slope]['max'] + 1,
                                                              len(extent[slope]),
                                                             )
                            cell.minmax[slope]['max'] = min(*[mycell.minmax[slope]['max'] for mycell in valuecells_in_extent],
                                                              cell.value - cell.minmax[other_slope]['min'] + 1,
                                                              len(greywhite_extent[slope]),
                                                             )
                            if cell.minmax[slope] != pre:
                                loopchange = True
                                self.updated = True


    def fit_min_for_known_length(self, cell, slope):
        ''' for a cell with an extent of known length, check if one of the ends is black, and if so, extend the length '''
        current_extent, last_cells = self.get_extent_by_color(cell, slope)
        for heading, last_cell_of_current_extent in last_cells.items():
            first_cell_of_next_extent = last_cell_of_current_extent.jump_to_neighbour(heading)
            if first_cell_of_next_extent.is_black():
                last_valid_cell = last_cell_of_current_extent
                self.fit_min_cell(cell, heading, last_valid_cell)
                return True
        return False


    def is_path_to_border_or_loop(self, cell, incoming_cell, original_cell):
        ''' 2 if this cell is the original cell
            1 if we can reach a border by passing through this black cell
            0 otherwise
        '''
        if cell == original_cell:
            # we returned to the origin, so we made a loop:
                return 2
        if cell.is_black():
            if cell.is_bordercell():
                return 1
            for neighbour in cell.diagonal_neighbours():
                if neighbour == incoming_cell:
                    pass
                elif self.is_path_to_border_or_loop(neighbour, cell, original_cell):
                    return 1
        return 0


    def cell_makes_no_separate_zone(self, cell):
        ''' Check if making this grey cell white would break the rule: "Any White square can be reached from any other" '''
        assert(cell.is_grey())
        pathcount = 1 if cell.is_bordercell() else 0
        for neighbour in cell.diagonal_neighbours():
            pathcount += self.is_path_to_border_or_loop(neighbour, cell, cell)
        if pathcount > 1:
            if cell.make_white():
                self.updated = True


    def no_separate_zones(self):
        ''' Check if making any grey cell white would break the rule: "Any White square can be reached from any other" '''
        for cell in self.getallgreycells():
            self.cell_makes_no_separate_zone(cell)


    def trial_and_error_one_2degreescell(self, cell):
        ''' Some value cells have only two degrees of freedom.
            For those cells we can check whether making the grey cell at the end of one degree black would result in a too long extent in the other direction.
            If that is the case, the assumption that that cell if black is incorrect, so it must be white.
        '''
        ### TODO: probably find_max_values already covers the case where the two free degrees are on the same slope. if so, this function can be simplified. ###

        # collect for each heading the end cell, the border call  and whether it is ended by a black cell
        heading_info =[]
        for heading in headings:
            endcell = self.get_extent_end_by_heading(cell, heading, lambda x: x.is_white())
            border = endcell.jump_to_neighbour(heading)
            heading_info.append({'heading': heading, 'endcell': endcell, 'bordercell': border, 'black': border.is_black()})
        # verify that there are only two degrees of freedom
        free_headings = [_ for _ in heading_info if not _['black']]
        if len(free_headings) != 2:
            return

        for key in range(2):    # select each free heading in order and check what would happen if we would make the border cell at the other free heading black
            other_key = 1-key
            my_free_heading = free_headings[key]  # the heading we are going to test
            other_free_heading = free_headings[other_key]  # the one we assume to be black

            my_slope, other_slope = my_free_heading['heading'][0], my_free_heading['heading'][0].inverse()
            my_slope_length = self.get_distance(*[_['endcell'] for _ in heading_info if _['heading'][0] == my_slope]) + 1
            other_slope_length = self.get_distance(*[_['endcell'] for _ in heading_info if _['heading'][0] == other_slope]) + 1
            myslope_required_length = cell.value - other_slope_length + 1
            length_to_check = myslope_required_length - my_slope_length

            # check if our assumption results in a violation: no required cell should be black,
            # and at the end there should be the option to close the extent, so that cell should still be grey or black.
            # if these conditions are violated, we know that our assumption of a black cell was wrong, so it must be white
            neighbour = my_free_heading['bordercell']
            for count in range(length_to_check):
                if neighbour.is_black():
                    other_free_heading['bordercell'].make_white()
                    self.updated = True
                    return
                neighbour = neighbour.jump_to_neighbour(my_free_heading['heading'])
            else:
                if neighbour.is_white():
                    other_free_heading['bordercell'].make_white()
                    self.updated = True
                    return


    def trial_and_error_2degreescells(self):
        # this applies to value cells that have only two degrees of freedom.
        # We can make a guess where one degree is black, and check if the other degree would be compliant.
        for cell in  self.getallvaluecells(solution_states = [0,1]):
            self.trial_and_error_one_2degreescell(cell)


    def prevent_overreach(self):
        ''' if making a grey cell white would result in an extent that is too long, then that cell must be black '''
        # TODO there is opportunity to make this faster, as self.get_extent_by_color now gives the ends ??
        for cell in self.getallgreycells():
            for slope in slopes:
                    neighbours = [ cell.jump_to_neighbour((slope, _)) for _ in directions ]
                    extent = [ cell ]
                    for neighbour in neighbours:
                        if neighbour.is_white():
                            extent.extend(self.get_extent_by_color(neighbour, slope)[0])

                    if min([mycell.minmax[slope]['max'] for mycell in extent]) < len(extent):
                        if cell.make_black():
                            self.surround_black_cell(cell)
                            self.updated = True


    def surround_black_cell(self, cell):
        ''' surround a black cell with white cells horizontally and vertically '''
        assert(cell.is_black())
        for heading in headings:
            whitecell = cell.jump_to_neighbour(heading)
            if whitecell.make_white():
                self.updated = True


    def surround_solved_cell(self, cell):
        ''' surround the extents for a solved cell with black cells
        '''
        assert(cell.is_solved())
        for heading in headings:
            bordercell = self.get_last_cell_by_color(cell, heading, nextcell = True)
            if bordercell.make_black():
                self.surround_black_cell(bordercell)
                self.updated = True


    def surround_solved_cells(self):
        for cell in self.getallvaluecells():
            if cell.is_solved():
                self.surround_solved_cell(cell)

    def fit_min_cell(self, cell, heading, last_valid_cell):
        ''' starting from cell, and going towards heading, last_valid_cell is the last cell can still be part of the extent of cell
            make cells in the inverse heading white to be able to fit in at least the minimum needed for the extent
        '''
        distance = abs(self.get_distance(cell, last_valid_cell))
        minlen = cell.minmax[heading[0]]['min']
        if minlen > distance:
            neighbour = cell
            inv_heading = inverse_heading(heading)
            for count in range(minlen - distance - 1):
                neighbour = neighbour.jump_to_neighbour(inv_heading)
                if neighbour.make_white():
                    self.updated = True

    def fit_min(self):
        for cell in self.getallvaluecells(solution_states = [0]):
            for slope in slopes:
                _, bordercells = self.get_greywhite_extent(cell, slope)
                for heading, bordercell in bordercells.items():
                    self.fit_min_cell(cell, heading, bordercell)



    def solve(self, verbose = False):
        methods = [ self.find_max_values, self.fit_min, self.prevent_overreach, self.no_separate_zones, self.trial_and_error_2degreescells]
        try:
            count = 0
            while not self.is_solved():
                count += 1
                if count in []:
                    pdb.set_trace()
                if self.updated:
                    self.updated = False
                    self.methodqueue = deque(methods)
                method = self.methodqueue.popleft()
                if verbose:
                    print(count, method.__name__)
                method()
                if verbose:
                    if self.updated:
                        print(self.nbr_unsolved_cells)
                        print(self)

        except IndexError:
            pass
        return self.is_solved()


    def __str__(self):
        result=''
        for key, row in enumerate(self.raster):
            rowstring=''
            for cell in row:
                rowstring += str(cell)
            result += rowstring + '\n'
        return result


import requests
import re

def get_puzzle(date, size):
    ''' Download a puzzle from the brainbashers site '''
    def getvariable(htmltext, varname):
        pattern = varname+'\s*=\s*[\']*([^";]+)[\']*;'
        result = re.search(pattern, htmltext)
        if result:
            return result.group(1)
        return None

    def geturl(date, size):
        return 'https://brainbashers.com/showrange.asp?date='+str(date)+'&size='+str(size)

    def nextcell(string):
        as_array = string.split(',')
        for char in as_array:
            yield char

    print(f'Getting puzzle for date: {date} & size: {size}')
    r = requests.get(geturl(date, size))
    lnwidth = int(getvariable(r.text, 'lnwidth'))
    lnheight = int(getvariable(r.text, 'lnheight'))
    lcpuzzle = getvariable(r.text, 'lcpuzzle')

    puzzle = rangeraster(lnheight, lnwidth, date)
    gen = nextcell(lcpuzzle)
    for mycell in puzzle.allcells():
        c = next(gen)
        if c == '*':
            pass
        elif c == 'W':
            mycell.make_white()
        elif c == 'B':
            mycell.make_black()
        else:
            mycell.set_value(int(c))
    return puzzle


def solve_brainbasher_range(date, size, verbose =  False):
    ''' solve a puzzle for given date and size
    '''
    puzzle = get_puzzle(date, size)
    solved = puzzle.solve(verbose)


def main():
    '''
    date='0101'
    sizes = ['4x3','5x4','6x5','7x6','8x7','10x8','12x9','15x10']
    #sizes = ['7x6']
    for size in sizes:
        solve_brainbasher_range(date, size, verbose = True)
    quit()
    '''
    today = datetime.date.today()
    otoday = datetime.date.toordinal(today)
    start = datetime.date(today.year, 1, 1)
    ostart = datetime.date.toordinal(start)
    sizes = ['4x3','5x4','6x5','7x6','8x7','10x8','12x9','15x10']
    #sizes = ['15x10']
    puzzles = []
    for odate in range(ostart, otoday+1):
        thedate = datetime.date.fromordinal(odate)
        date = f'{thedate.month:02}{thedate.day:02}'
        for size in sizes:
            puzzles.append(get_puzzle(date, size))
    t0 = time.process_time()
    for puzzle in puzzles:
        puzzle.solve(verbose=False)
        print('date: ', puzzle.name, 'width: ', puzzle.width, 'height: ', puzzle.height)
        if not puzzle.is_solved():
            print(puzzle)
    t1 = time.process_time()
    print(f'time for solving {len(puzzles)} puzzles = {t1-t0}')

if __name__ == '__main__':
    main()
