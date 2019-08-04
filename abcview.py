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

class Cell(BrainbasherCell):
    """ a single cell in abcview """

    def __init__(self, row, col):
        super().__init__(row, col, 'ABCXX', Color('grey'))


    def __str__(self):
        viewstr = ''
        for letter in 'ABC':
            viewstr += letter if letter in self.value else '_'
        viewstr += {0:'__', 1:'X_', 2:'XX'}[self.count_X()]
        viewstr += ' '
        return colored(viewstr, *self.color.termcolors())

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
            raise ContradictionFound()


    def autocolor(self):
        if len(self.value) == 1:
            self.color = Color('white')

    def is_solved(self):
        return len(self.value) == 1


    def count_X(self):
        return len([_ for _ in self.value if _ == 'X'])


    def set_value(self, possible_letters):
        self.value = possible_letters if possible_letters else 'ABCXX'
        self.autocolor()


    def remove_letters(self, letters):
        for letter in letters:
            assert(letter in 'ABC')
            self.value = self.value.replace(letter,'')
        if self.value == 'XX':
            self.value = 'X'
        self.autocolor()


    def set_max_X(self, nbr_X_allowed):
        if len(self.value) > 1:
            nbr_to_remove = self.count_X() - nbr_X_allowed
            if nbr_to_remove > 0:
                self.value = self.value.replace('X','',nbr_to_remove)
        self.autocolor()


class Raster (BrainbasherRaster):
    def _newcell(self,row, col):
        return Cell(row,col)

    def __init__(self, nbr_rows, nbr_cols, identity):
        super().__init__(nbr_rows, nbr_cols, identity)
        self.hints = []

    def setvalues(self, values: list):
        newvalues = ['ABCXX' if _ == '.' else _ for _ in values]
        super().setvalues(newvalues)

    def sethints(self, top, bottom, left, right):
        hints = []
        for hint_input in [left, top, right, bottom]:  # this must match the order of make_virtualrow
            hints.extend(hint_input)
        assert(len(hints) == 20)
        self.hints = hints

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
        except ContradictionFound:
            self.undo_guess(white_guessresults)
            # our guess (white) was not correct, so we assume black
            # if this also throws, it means that something was wrong before we were called,
            # so we don't catch that exception but let the exception bubble up
            print('guess: ', row, ', ', col, '=black')
            self.make_black(row, col, guessresults)
            if not self.set_closing_cells_white_until_solved(guessresults):
                self.guess(guessresults)


    def make_virtualrow(self, idx):
            return super().make_virtualrow(idx), self.hints[idx]


    def countcolors(self, virtualrow):
        row, col, line = virtualrow
        colors = ('black','white')
        result = []
        for color in colors:
            result.append(len([_ for _ in line if _.color is color]))
        return result



    def restrict_by_hint(self, virtualrow, hint):
        """ If a hint is given, then:
            * The first cell which is not all X should contain no other letters than the hint.
            * looking from the back, the hint cannot be present in a cell, until we've seen both other letters
        """
        if not hint:
            return

        otherletters= 'ABC'.replace(hint,'')
        # ensure the hint is the first letter seen
        for cell  in virtualrow:
            cell.remove_letters(otherletters)
            if hint in cell.value:
                    break
        # ensure other letters can be placed after hint
        cell_cleaned = 0
        for cell in reversed(virtualrow):
            cell.remove_letters(hint)
            letters_seen = [ _ for _ in otherletters if _ in cell.value]
            if len(letters_seen) == 0:
                continue
            elif len(letters_seen) == 1:
                otherletters.replace(letters_seen[0],'')
            cell_cleaned += 1
            if cell_cleaned == 2:
                break


    def remove_found_letters(self, virtualrow, hint):
        """ for all cells of which we know the lettervalue, we can eliminate that letter from the row
            also, as soon as we have seen any letter, we know that the cels after that letter cannot contain the hint
        """
        known_letter_seen = False
        for cell in virtualrow:
            if known_letter_seen and hint:
                cell.remove_letters(hint)
            if cell.value in 'ABC':
                known_letter_seen = True
                for othercell in [_ for _ in virtualrow if _ is not cell]:
                    othercell.remove_letters(cell.value)

    def count_X(self, virtualrow):
        return len([_ for _ in virtualrow if _.value == 'X'])

    def remove_found_X(self, virtualrow):
        """ Adapt the number of possible X in each cell to the number already found """
        nbr_X_allowed = 2 - self.count_X(virtualrow)
        if nbr_X_allowed < 2:
            for cell in virtualrow:
                cell.set_max_X(nbr_X_allowed)

    def unique_letter_in_row(self, virtualrow):
        for letter in 'ABC':
            cells_with_letter = [_ for _ in virtualrow if letter in _.value]
            if len(cells_with_letter) == 1:
                cells_with_letter[0].set_value(letter)

    def solve(self):
        self.print_me()
        previous_unsolved = None
        unsolved = self.nbr_unsolved_cells()
        while unsolved != previous_unsolved:
            previous_unsolved = unsolved
            for row in range(self.nbr_rows*4):
                virtualrow, hint = self.make_virtualrow(row)
                self.remove_found_letters(virtualrow,hint)
                self.remove_found_X(virtualrow)
                self.restrict_by_hint(virtualrow, hint)
                self.unique_letter_in_row(virtualrow)

            unsolved = self.nbr_unsolved_cells()
        if unsolved == 0:
            return self
        else:
            print('could not solve this:')
            self.print_me()
            #self.guess(guessresults = [])
            #self.print_me()
            return self

    def print_me(self):
        hints = [_ if _ else ' ' for _ in self.hints]
        filler = ' '*6
        line = filler
        for col in range(5,10):
            line += '  '+hints[col]+'   '
        print(line)
        for row in range(self.nbr_rows):
            line = '  '+hints[row]+'   ' + self.str_by_row(row) + '  '+hints[10+row]
            print(line)
        line = filler
        for col in range(15,20):
            line += '  '+hints[col]+'   '
        print(line)


def puzzle_from_html(html) -> Raster:
    puzzle_string = get_javascript_variable(html, 'lcq')
    if not puzzle_string:   # received html was not a valid puzzle
        return None
    puzzle_top = get_javascript_variable(html, 'lctop')
    puzzle_bottom = get_javascript_variable(html, 'lcbottom')
    puzzle_left = get_javascript_variable(html, 'lcleft')
    puzzle_right = get_javascript_variable(html, 'lcright')
    puzzle_size = 5
    puzzle_id = get_javascript_variable(html, 'lcpuzzletext')
    puzzle = Raster(puzzle_size, puzzle_size, puzzle_id)
    puzzle.setvalues(puzzle_string)
    puzzle.sethints(*[ [None if _ == 'z' else _ for _ in hints] for hints in [puzzle_top, puzzle_bottom, puzzle_left, puzzle_right]])
    return puzzle



def puzzle_urls(whichs=[1,2,3,4,5,6]):
    """ generate urls for all possible abcview puzzles
    """
    tomorrow = datetime.today()+timedelta(days=1)
    for urldate in daterange(datetime(datetime.today().year,7,26), tomorrow):
            for which in whichs:
                yield f'https://brainbashers.com/showabcview.asp?date={urldate.month:02}{urldate.day:02}&which={which}'


async def main(loop):
    async with ClientSession(loop=loop) as session:
        for url in puzzle_urls():
            solution = await solve_brainbasher_task(session, url,puzzle_from_html)
            if solution and not solution.is_solved():
                print(solution.identity)
                solution.print_me()

def testmain():
        import requests
        url = 'https://brainbashers.com/showabcview.asp?date=0727&which=4'
        response = requests.get(url)
        puzzle = puzzle_from_html(response.text)
        puzzle.solve()
        puzzle.print_me()

if __name__ == '__main__':
    #testmain()
    #quit()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))
