#! /bin/env python

import pdb
import asyncio
import async_timeout
from datetime import datetime, timedelta
from aiohttp import ClientSession
from termcolor import colored
from collections import defaultdict
from re import search as regex_search

class Cell:
    """ a single cell in a 3-in-a-row """

    # convert 'logical' colors in term colors
    colordict = {'white': ('grey','on_white'),
                 'black': ('yellow','on_grey'),
                 'grey': ('blue', 'on_red')
                 }

    invertedcolordict = {'white': 'black',
                 'black': 'white',
                 'grey': 'grey',
                 }
    def __init__(self):
        self.value = ' '
        self.color = 'grey'

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
        assert(color in self.colordict.keys())
        if self.color == 'grey' or not check_consistency:
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

    def setcolors(self, colors: list):
        for row in range(self.nbr_rows):
            for col in range(self.nbr_cols):
                self.raster[row][col].color = colors[row * self.nbr_cols + col]

    def setvalues(self, values: list):
        for row in range(self.nbr_rows):
            for col in range(self.nbr_cols):
                self.raster[row][col].value = values[row * self.nbr_cols + col]

    def nbr_unsolved_cells(self):
        return len([1 for row in range(self.nbr_rows) for col in range(self.nbr_cols) if self.raster[row][col].color == "grey"])

    def is_solved(self):
        return not any([self.raster[row][col].color == "grey" for row in range(self.nbr_rows) for col in range(self.nbr_cols)])

    def set_color(self, row, col, color, guessresults = []):
        ''' Set the color of a cell. For the purpose of backtracking, remember add it to the list guessresult '''
        if self.raster[row][col].set_color(color):
            guessresults.append((row,col))
            print(f'set ({row},{col}) {color}')
            return True
        return False

    def set_color_if_valid(self, row, col, color, guessresults = []):
        ''' Set the color of a cell if the indexes are valid. For the purpose of backtracking, remember add it to the list guessresult '''
        if row < 0 or col < 0:
            return False
        try:
            return self.set_color(row, col, color, guessresults)
        except IndexError:
            return False

    def is_border_cell(self, row, col):
        """ True if the cell is at the edge of the board
        """
        return  row == 0 or row == self.nbr_rows - 1 or col == 0 or col == self.nbr_cols - 1

    def get_any_grey_cell(self):
        """ return any cell from the raster that is still grey
        """
        for row in range(self.nbr_rows):
            for col in range(self.nbr_cols):
                if self.raster[row][col].color == "grey":
                    yield row,col


    def undo_guess(self, guessresults):
        """ backtrack a recursion given by the list guessresult
        """
        for row, col in guessresults:
            self.raster[row][col].set_color("grey", check_consistency = False)


    def guess(self, guessresults):
        """ solve by  brute force (recursion)
        """
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


    def cellid(self, virtualrow, idx):
        """ get the raster row and column of a cell by combining the virtualrow and an index
        """
        row, col, cells = virtualrow
        row = row if row is not None else idx
        col = col if col is not None else idx
        return row, col

    def surround_duo(self, virtualrow):
        """ adjacent cells of the same color should be surrounded by neighbours of a different color
        """
        row, col, line = virtualrow
        for idx in range(len(line)-1):
            color = line[idx].color
            if color is not 'grey' and color == line[idx + 1].color:
                othercolor = Cell.invertedcolordict[color]
                self.set_color_if_valid(*self.cellid(virtualrow, idx - 1), othercolor)
                self.set_color_if_valid(*self.cellid(virtualrow, idx + 2), othercolor)

    def fill_in_between(self, virtualrow):
        """ cells of the same color with one grey cell in between shoudl be separated by a cell with the opposite color
        """
        row, col, line = virtualrow
        for idx in range(len(line)-2):
            color = line[idx].color
            if color is not 'grey' and color == line[idx+2].color:
                othercolor = Cell.invertedcolordict[color]
                self.set_color_if_valid(*self.cellid(virtualrow, idx + 1), othercolor)

    def complete_virtualrow_with_color(self, virtualrow, color):
        row, col, line = virtualrow
        for idx in range(len(line)):
            cellcolor = line[idx].color
            if cellcolor is 'grey':
                self.set_color(*self.cellid(virtualrow, idx), color)

    def complete_half_completed_row(self, virtualrow):
        row, col, line = virtualrow
        nbr_black, nbr_white = self.countcolors(virtualrow)
        if nbr_black == len(line)/2:
            self.complete_virtualrow_with_color(virtualrow, 'white')
        elif nbr_white == len(line)/2:
            self.complete_virtualrow_with_color(virtualrow, 'black')

    def make_virtualrow(self, row, col):
        virtualrow = row, None, None if row is None else [self.raster[row][_] for _ in range(self.nbr_cols)]
        virtualcol = None, col, None if col is None else [self.raster[_][col] for _ in range(self.nbr_rows)]
        return virtualrow, virtualcol

    def get_grey_regions(self, virtualrow):

        def border_info(first_color_seen, last_color_seen):
            return first_color_seen, last_color_seen, len([_ for _ in [first_color_seen, last_color_seen] if _ == 'grey'])

        row, col, line = virtualrow
        regions = []
        currentregion = []
        region_start_idx, first_color_seen = 0, line[0].color
        last_color_seen = None
        for idx, cell in enumerate(line):
            if cell.color is 'grey':
                currentregion.append(cell)
            else:
                if last_color_seen is 'grey':
                    currentregion.append(cell)
                    regions.append((currentregion, region_start_idx,*border_info(first_color_seen,cell.color)))
                region_start_idx, first_color_seen = idx, cell.color
                currentregion = [cell]
            last_color_seen = cell.color
        if last_color_seen is 'grey':
            regions.append((currentregion, region_start_idx,*border_info(first_color_seen,last_color_seen)))
        return regions

    def unicolor_region_allowed(self, region, color):
        """ return for this region the minimal number of colors that we know it must contain (even though we may not know where)
        """
        def disallowedcolor(startcolor, endcolor):
            return startcolor if startcolor is not 'grey' else endcolor

        line, startidx, startcolor, endcolor, greybordercount = region
        if len(line) == 2:
            return (0,0)
        elif len(line) > 4:
            return (1,1)
        elif len(line) == 3:
            # we assume here that cases such as white/grey/white have already been filled, but if this is not the case,
            # it does not hurt. the region will be treated as 'we don't know if this cell must be filled'
            if greybordercount == 0: # one grey between two cells could be anything
                return (0,0)
            else:    # two grey cells at the the edge of the board bordered by one of color, have at least one of the opposite color: |__B must contain 1 W
                return (1,0) if disallowedcolor(startcolor, endcolor) is 'white' else (0,1)
        else:    # len(line) == 4
            if greybordercount == 0:
                if startcolor != endcolor:  # two greys between a black and a white must be one of each
                    return (1,1)
                else:   # same color on both ends, so we need at least one of the opposite color
                    return (1,0) if startcolor is 'white' else (0,1)
            else:   # three greys require at least one of each
                return (1,1)

    def partially_complete_by_region(self, virtualrow):
        """ Try to determine for all regions where at least one color is present. If all of these add up to the max allowed for that color,
            then all other regions have the opposite color and can be filled
        """

        def get_fillable_regions_by_color(regions, fillcolor, oppositecolorcount):
            color_fillable_regions = []
            minimal_assigned_color = 0
            if oppositecolorcount == max_of_color:
                color_fillable_regions = regions
            else:
                for region in regions:
                    nbr_black, nbr_white = self.unicolor_region_allowed(region) # TODO it is better to pass color as a function to unicolor, as it avoids the next if
                    if fillcolor == 'black':
                        region_should_not_contain_any = nbr_white
                    if fillcolor == 'white':
                        region_should_not_contain_any = nbr_black
                    if  region_should_not_contain_any == 0:
                        color_fillable_regions.append(region)
                    else:
                        minimal_assigned_color += region_should_not_contain_any
            return color_fillable_regions, minimal_assigned_color

        def filter_X_C_X(fillable_regions, color):
            minimal_assigned_opposite_color = 0
            if fillable_regions:
                fillable = [ True for _ in fillable_regions ]
                for idx in range(len(fillable_regions) - 1):
                    first_region_line,_,_,endcolor,first_greybordercount = fillable_regions[idx]
                    next_region_line,_,_,_,next_greybordercount  = fillable_regions[idx+1]
                    first_greycount = len(first_region_line) - 2 + first_greybordercount
                    next_greycount = len(next_region_line) - 2 + next_greybordercount
                    if first_region_line[-1] is next_region_line[0] and first_greycount == 1 and next_greycount == 1: # adjacent short regions
                            if endcolor == color:
                                minimal_assigned_opposite_color += 1
                                fillable[idx], fillable[idx+1] = False, False
                filtered_fillable_regions = [ _[0] for _ in zip(fillable_regions, fillable) if _[1]]
            else:
                filtered_fillable_regions = []
            return filtered_fillable_regions, minimal_assigned_opposite_color


        def fill_fillable_regions(fillable_regions, fillcolor):
            for region in fillable_regions:
                line, startidx,_,_,_ = region
                for idx,cell in enumerate(line):
                    if cell.color is 'grey':
                        self.set_color(*self.cellid(virtualrow,startidx+idx),fillcolor)


        def fill_longregions(longregions, fillcolor):
            for region in longregions:
                line, startidx, startcolor, endcolor, greybordercount = region
                verylong = (len(line) + greybordercount > 5) # it is a 'very' long grey area, so we cannot put our single separator at the sides
                if startcolor == fillcolor or verylong:
                    fillindex = len(line) - (1 if endcolor is 'grey' else 2)
                    self.set_color(*self.cellid(virtualrow,startidx+fillindex),fillcolor)
                if endcolor == fillcolor or verylong:
                    fillindex = 0 if startcolor is 'grey' else 1
                    self.set_color(*self.cellid(virtualrow,startidx+fillindex),fillcolor)


        row, col, virtline = virtualrow
        max_of_color = len(virtline)/2
        regions = self.get_grey_regions(virtualrow)
        if not regions: # the line is already fully solved
            return
        known_blacks, known_whites = self.countcolors(virtualrow)
        whitefillable_regions, minimal_blacks = get_fillable_regions_by_color(regions, 'white', known_blacks)
        blackfillable_regions, minimal_whites = get_fillable_regions_by_color(regions, 'black', known_whites)

        # The folllowing assumes the 'fill' check has already run
        # This should not be run for both black and white at the same time
        # Any combined region X_B_X also contains at least one W, so these should be removed too from the fillable regions
        # and one color should be counted
        # however, this must be done separately for white and black because 
        # otherwise X_B_W_X would result in all three _ being taken away as fillable, whereas if study black and white separately,
        # you get X?B?WFX and XFB?W?X  (? means there must be one of that color there, F means fillable
        whitefillable_regions, assigned_blacks = filter_X_C_X(whitefillable_regions, 'white')
        minimal_blacks += assigned_blacks
        blackfillable_regions, assigned_whites = filter_X_C_X(blackfillable_regions, 'black')
        minimal_whites += assigned_whites

        # we just detected for each region how many items of each color it must contain at least.
        # if the sum of these with the already known items is equal  to the maximum,
        # then all regions that don't require that color have the opposite color.
        if known_blacks + minimal_blacks == max_of_color:
            fill_fillable_regions(whitefillable_regions, 'white')
        if known_whites + minimal_whites == max_of_color:
            fill_fillable_regions(blackfillable_regions, 'black')

        # if we know we have only one black resp; white color we can place for the row, then if there is a long region (>= 3 greys)
        # it must be placed in that long region, and it must be not be placed on the other side of a border with fillcolor
        # e.g B___, and we can place only one W (fillcolor=B), then we cannot place it B__W, so we must have B__B
        # or W___W, then and we can only place one B (fillcolor=W), then it cannot be WB__W or W__BW
        longregions = [ _ for _ in regions if  len(_[0]) + _[4] > 4 ]
        # check if we can only place one item in each long region
        if known_blacks + minimal_blacks - len(longregions) == max_of_color - 1:
            fill_longregions(longregions, 'white')
        if known_whites + minimal_whites - len(longregions) == max_of_color - 1:
            fill_longregions(longregions, 'black')

        '''
        if fillcolor:
            for region in longregions:
                line, startidx, startcolor, endcolor, greybordercount = region
                verylong = (len(line) + greybordercount > 5) # it is a 'very' long grey area, so we cannot put our single separator at the sides
                if startcolor == fillcolor or verylong:
                    fillindex = len(line) - (1 if endcolor is 'grey' else 2)
                    self.set_color(*self.cellid(virtualrow,startidx+fillindex),fillcolor)
                if endcolor == fillcolor or verylong:
                    fillindex = 0 if startcolor is 'grey' else 1
                    self.set_color(*self.cellid(virtualrow,startidx+fillindex),fillcolor)
        '''


    def countcolors(self, virtualrow):
        row, col, line = virtualrow
        colors = ('black','white')
        result = []
        for color in colors:
            result.append(len([_ for _ in line if _.color is color]))
        return result


    def solve(self):
        self.print_me()
        previous_unsolved = None
        unsolved = self.nbr_unsolved_cells()
        while unsolved != previous_unsolved:
            previous_unsolved = unsolved
            for row in range(self.nbr_rows):
                for virtualrow in self.make_virtualrow(row,row):
                    #print('surround', virtualrow[0:2])
                    self.surround_duo(virtualrow)
                    #self.print_me()
                    #print('fill', virtualrow[0:2])
                    self.fill_in_between(virtualrow)
                    #self.print_me()
                    #self.complete_half_completed_row(virtualrow) 
                    #print('region', virtualrow[0:2])
                    self.partially_complete_by_region(virtualrow)
                    #self.print_me()
                    #print('unsolved ', self.nbr_unsolved_cells())
            unsolved = self.nbr_unsolved_cells()
        self.print_me()
        if unsolved == 0:
            return self
        else:
            print('could not solve this:')
            self.print_me()
            #self.guess(guessresults = [])
            #self.print_me()
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
    color = {'B': 'black', 'W': 'white', '-': 'grey'}
    puzzle.setcolors([color[_] for _ in puzzle_string])
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


def puzzle_urls(sizes=[6,8,10,12,14], levels=[1,2], include_weekly = True):
    """ generate urls for all possible hitory puzzles
        weekly puzzles (size 12) only works for the last week
    """
    tomorrow = datetime.today()+timedelta(days=1)
    for urldate in daterange(datetime(datetime.today().year,6,10), tomorrow):
        for level in levels:
            for size in sizes:
                yield f'https://brainbashers.com/show3inarow.asp?date={urldate.month:02}{urldate.day:02}&size={size}&diff={level}'
    for urldate in daterange(datetime.today()-timedelta(days=7), tomorrow):
        level, size = 1, 12
        yield f'https://brainbashers.com/show3inarow.asp?date={urldate.month:02}{urldate.day:02}&size={size}&diff={level}'



async def main(loop):
    async with ClientSession(loop=loop) as session:
        for url in puzzle_urls():
            solution = await solve_brainbasher_task(session, url)
            if solution and not solution.is_solved():
                print(solution.identity)
                solution.print_me()

def testmain():
        import requests
        url = 'https://brainbashers.com/show3inarow.asp?date=0802&diff=2&size=18'
        response = requests.get(url)
        puzzle = puzzle_from_html(response.text)
        puzzle.solve()
        puzzle.print_me()

if __name__ == '__main__':
    #testmain()
    #quit()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))
