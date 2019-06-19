#! /bin/env python

import pdb
from enum import Enum, IntEnum
from typing import List
from collections import deque

class cellstate(Enum):
    forbidden = -1
    unknown = 0
    occupied = 1

class direction(IntEnum):
    north = 0
    east = 1
    south = 2
    west = 3

    def inverse(self):
        ''' north becomes south, east becomes west, ... '''
        # this implementation relies on the values, order and number of directions !
        return direction((self.value + 2) % 4)

class cell:
    def __init__(self, state = cellstate.unknown, north: cellstate = cellstate.unknown, east: cellstate = cellstate.unknown, south: cellstate = cellstate.unknown, west: cellstate = cellstate.unknown, id = None):
        self.id = id
        self.state = state
        self.connections = [ north, east, south, west ]
        self.verifyconnections()

    def connectioncount(self, state: cellstate):
        ''' count the number of headings in a specific state '''
        return len([ _ for _ in self.connections if _ is state ])

    def verifyconnections(self):
        updated = []
        nbr_occupied  = self.connectioncount(cellstate.occupied)
        nbr_forbidden = self.connectioncount(cellstate.forbidden)
        if nbr_occupied > 0:
            self.state = cellstate.occupied
        if nbr_occupied == 2:
            for heading in direction:
                if self.connections[heading] == cellstate.unknown:
                    self.connections[heading] = cellstate.forbidden
                    updated.append(heading)
        if nbr_forbidden == 3:
            self.state = cellstate.forbidden
            for heading in direction:
                if self.connections[heading] == cellstate.unknown:
                    self.connections[heading] = cellstate.forbidden
                    updated.append(heading)
        if nbr_forbidden == 2 and self.state == cellstate.occupied:
            for heading in direction:
                if self.connections[heading] == cellstate.unknown:
                    self.connections[heading] = cellstate.occupied
                    updated.append(heading)
        return updated

    def set_connection(self, heading: direction, state: cellstate):
        self.connections[heading] = state
        return self.verifyconnections()

    def __repr__(self):
        result = ''
        try:
            result += f'{self.state}'
        except:
            pass
        try:
            result += f', {self.connections}'
        except:
            pass
        return result

    def __str__(self):
        def binary_converter(arr):
            total = 0
            for index, val in enumerate(reversed([1 if _ == cellstate.occupied else 0 for _ in self.connections])):
                total += (val * 2**index)
            return total


        if self.state == cellstate.unknown:
            return  '.'
        elif self.state == cellstate.forbidden:
            return 'x'
        else:
            lookup = binary_converter(self.connections)
            if lookup == 14:
                print(self.id)
            chardict = { 0: '+',
                         1: u'\u2574', 8: u'\u2575', 4: u'\u2576', 2: u'\u2577',
                         5: u'\u2550', 10: u'\u2551', 6: u'\u2554', 3: u'\u2557', 12: u'\u255A', 9: u'\u255D'
                       }
            return chardict[lookup]

class trackraster:
    def __init__(self, size: int, railsections_per_row: List[int], railsections_per_col: List[int]):
        self.updatequeue = deque()
        self.size = size
        self.railsections_per_row = railsections_per_row
        self.railsections_per_col = railsections_per_col
        self.raster = [ [cell()  for col in range(size)] for row in range(size)]
        for row in range(size):
            for col in range(size):
                self.updatecell(row,col,cell())
        self.updatequeue.clear()

    def get_neighbour(self, row, col, heading):
        ''' Get the neighbouring cell in the given direction.
            raise IndexError if no such cell exists
        '''
        try:
            if heading == direction.south:
                return self.raster[row + 1][col]
            elif heading == direction.east:
                return self.raster[row][col + 1]
            elif heading == direction.north:
                if row == 0:
                    raise IndexError
                return self.raster[row - 1][col]
            elif heading ==direction.west:
                if col == 0:
                    raise IndexError
                return self.raster[row][col - 1]
        except IndexError:
            return cell(*[cellstate.forbidden]*5, id=None)

    def updatecell(self, row, col, newcell):
        if not newcell.id:
            newcell.id=(row,col)
        for heading in direction:
            if newcell.connections[heading] is cellstate.unknown:
                newcell.set_connection(heading, self.get_neighbour(row, col, heading).connections[heading.inverse()])
        self.raster[row][col] = newcell
        for heading in direction:
            if newcell.connections[heading] is not cellstate.unknown:
                neighbour = self.get_neighbour(row, col, heading)
                if neighbour.id:
                    self.updatequeue.append((*neighbour.id, heading.inverse(), newcell.connections[heading]))

    def occupycell(self, row, col):
        if self.raster[row][col].state == cellstate.unknown:
            self.updatecell(row, col, cell(cellstate.occupied, *[cellstate.unknown]*4))

    def forbidcell(self, row, col):
        if self.raster[row][col].state == cellstate.unknown:
            self.updatecell(row, col, cell(*[cellstate.forbidden]*5))

    def update_connection(self, row, col, heading, state):
        cell = self.raster[row][col]
        updates = cell.set_connection(heading, state)
        for heading in updates:
            neighbour = self.get_neighbour(row, col, heading)
            if neighbour.id:
                self.updatequeue.append((*neighbour.id, heading.inverse(), cell.connections[heading]))

    def occupy_connection(self, row, col, heading):
        self.update_connection(row, col, heading, cellstate.occupied)

    def forbid_connection(self, row, col, heading):
        self.update_connection(row, col, heading, cellstate.forbidden)

    def set_between_cells(self, row1, col1, row2, col2, state):
        if col1 == col2:
            if row1 - row2 == 1:
                heading = direction.north
            elif row1 - row2 == -1:
                heading = direction.south
        elif row1 == row2:
            if col1 - col2 == 1:
                heading = direction.west
            elif col1 - col2 == -1:
                heading = direction.east
        self.update_connection(row1, col1, heading, state)
        self.update_connection(row2, col2, heading.inverse(), state)

    def complete_rows(self):
        for row in range(self.size):
            nbr_occupied = 0
            nbr_forbidden = 0
            for col in range(self.size):
                nbr_occupied += 1 if self.raster[row][col].state == cellstate.occupied else 0
                nbr_forbidden += 1 if self.raster[row][col].state == cellstate.forbidden else 0
            if nbr_occupied == self.railsections_per_row[row]:
                for col in range(self.size):
                    self.forbidcell(row, col)
            if nbr_forbidden == self.size - self.railsections_per_row[row]:
                for col in range(self.size):
                    self.occupycell(row, col)

        for col in range(self.size):
            nbr_occupied = 0
            nbr_forbidden = 0
            for row in range(self.size):
                nbr_occupied += 1 if self.raster[row][col].state == cellstate.occupied else 0
                nbr_forbidden += 1 if self.raster[row][col].state == cellstate.forbidden else 0
            if nbr_occupied == self.railsections_per_col[col]:
                for row in range(self.size):
                    self.forbidcell(row, col)
            if nbr_forbidden == self.size - self.railsections_per_col[col]:
                for row in range(self.size):
                    self.occupycell(row, col)

    def follow(self, row, col, heading):
        ''' given a certain cell at the end of a path, find the cell at the other end of the path '''
        cell = self.get_neighbour(row, col, heading)
        if cell.connectioncount(cellstate.occupied) == 1:
            # a 'regular' end of a trail
            return cell
        elif cell.connectioncount(cellstate.occupied) == 0:
            # a start or ending point
            return None
        else:
            comming_from = heading.inverse()
            for nextheading in direction:
                if cell.connections[nextheading] == cellstate.occupied and nextheading != comming_from:
                    return self.follow(*cell.id, nextheading)

    def noloops(self):
        ''' for a given cell at the end of a path, if we can follow the path and reach a neighbour, then the connection to that neighbour must be forbidden '''
        for row in range(self.size):
            for col in range(self.size):
                cell = self.raster[row][col]
                if cell.state == cellstate.occupied and cell.connectioncount(cellstate.occupied) == 1:
                    neighbours = [self.get_neighbour(row, col, heading) for heading in direction if cell.connections[heading] == cellstate.unknown]
                    otherend = self.follow(row, col, [heading for heading in direction if cell.connections[heading] == cellstate.occupied ][0])
                    if otherend in neighbours:
                        self.set_between_cells(*cell.id, *otherend.id, cellstate.forbidden)

    def noshortcuts(self):
        ''' if a cell at the end of a path reaches the exit of the raster by following the path, and its neighbour does the same, then this path should be closed to solve the puzzle, unless there are still other unsolved railsegments '''
        for row in range(self.size):
            for col in range(self.size):
                cell = self.raster[row][col]
                if cell.state == cellstate.occupied and cell.connectioncount(cellstate.occupied) == 1:
                    if not self.follow(row,  col, [heading for heading in direction if cell.connections[heading] == cellstate.occupied][0]):
                        neighbours = [self.get_neighbour(row, col, heading) for heading in direction if cell.connections[heading] == cellstate.unknown]
                        for neighbour in neighbours:
                            if neighbour.state == cellstate.occupied and neighbour.connectioncount(cellstate.occupied) == 1:
                                if not self.follow(*neighbour.id, [heading for heading in direction if neighbour.connections[heading] == cellstate.occupied ][0]):
                                    if self.has_max_open_segments(2):
                                        self.set_between_cells(*cell.id, *neighbour.id, cellstate.occupied)
                                    else:
                                        self.set_between_cells(*cell.id, *neighbour.id, cellstate.forbidden)


    def has_max_open_segments(self, max_open_segments: int):
        ''' check if the number of railsegments with unknown connections is less than or equal to the given maximum '''
        count = 0
        for row in range(self.size):
            for col in range(self.size):
                cell = self.raster[row][col]
                if cell.state == cellstate.occupied and cell.connectioncount(cellstate.occupied) != 2:
                    count += 1
                    if count > max_open_segments:
                        return False
        return True

    def propagate_changes(self):
        try:
            while True:
                row, col, heading, state = self.updatequeue.popleft()
                updates = self.update_connection(row, col, heading, state)
        except IndexError:
            pass # empty queue


    def issolved(self):
        return self.has_max_open_segments(0)

    def solve(self, verbose = False):
        methods = [ self.propagate_changes, self.complete_rows, self.noloops, self.noshortcuts ]
        try:
            while True:
                if self.updatequeue:
                    methodqueue = deque(methods)
                    method = methodqueue.popleft()
                    method()
                    if verbose:
                        print(method.__name__)
                        print(self)
                else:
                    method = methodqueue.popleft()
                    method()
                    if verbose:
                        print(method.__name__)
                        print(self)
        except IndexError:
            pass
        return self.issolved()

    def __str__(self):
        result = '  ' + str(self.railsections_per_col) + '\n'
        for key, row in enumerate(self.raster):
            rowstring = f'{self.railsections_per_row[key]:2}'
            for cell in row:
                rowstring += str(cell)
            result += rowstring + '\n'
        return result


import requests
import re

def solve_brainbasher(date, size, variant, verbose =  False):

    def getvariable(htmltext, varname):
        pattern = varname+'\s*=\s*["]*([^";]+)["]*;'
        result = re.search(pattern, htmltext)
        if result:
            return result.group(1)
        return None

    def geturl(date, size, variant):
        return 'https://brainbashers.com/showtracks.asp?date='+str(date)+'&type='+variant+'&size='+str(size)

    def rowinfolist(string):
        return [ ord(char) - ord('A') + 1 for char in string ]

    def cellfromchar(char):
        if char == '-':
            return cell()
        elif char == '3':
            return cell(cellstate.occupied, east=cellstate.occupied, north=cellstate.occupied)
        elif char == '5':
            return cell(cellstate.occupied, east=cellstate.occupied, west=cellstate.occupied)
        elif char == '9':
            return cell(cellstate.occupied, east=cellstate.occupied, south=cellstate.occupied)
        elif char == '6':
            return cell(cellstate.occupied, north=cellstate.occupied, west=cellstate.occupied)
        elif char == 'a':
            return cell(cellstate.occupied, north=cellstate.occupied, south=cellstate.occupied)
        elif char == 'c':
            return cell(cellstate.occupied, west=cellstate.occupied, south=cellstate.occupied)

    def nextcell(string):
        for char in string:
            yield cellfromchar(char)

    r = requests.get(geturl(date, size, variant))
    lctop = getvariable(r.text, 'lctop')
    lcright = getvariable(r.text, 'lcright')
    size = int(getvariable(r.text, 'lnsize'))
    lcgivens = getvariable(r.text, 'lcgivens')

    puzzle = trackraster(size, rowinfolist(lcright), rowinfolist(lctop))
    gen = nextcell(lcgivens)
    for row in range(size):
        for col in range(size):
            c = next(gen)
            puzzle.updatecell(row, col, c)
    if not puzzle.solve(verbose):
        print(date, size, variant)
        print(puzzle)

import datetime

def main():
    '''
    date='0301'
    size=10
    variant='B'
    solve_brainbasher(date, size, variant, verbose = True)
    quit()
    '''
    today = datetime.date.today()
    otoday = datetime.date.toordinal(today)
    start = datetime.date(today.year, 1, 1)
    ostart = datetime.date.toordinal(start)
    for size in range(5,10):
        for odate in range(ostart, otoday):
            for variant in ['A','B']:
                thedate = datetime.date.fromordinal(odate)
                date = f'{thedate.month:02}{thedate.day:02}'
                solve_brainbasher(date, size, variant)

if __name__ == '__main__':
    main()
