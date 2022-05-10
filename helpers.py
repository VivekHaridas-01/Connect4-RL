import numpy as np

def put_new_piece(grid, col, piece, config):

    found = False
    updated_grid = grid.copy()
    # Iterate through each row of the column from the bottom
    for row in range(config.rows - 1, -1, -1):
        # Find the the row without a piece already in it
        if not updated_grid[row][col]:
            found = True
            break
    
    if not found:
        print("[*] Column is full!")
        return

    # Place piece in the right spot
    updated_grid[row][col] = piece
    return updated_grid

def get_search_range(search_type, grid, row, column):
    
    # Create specific search ranges for each search direction
    if search_type == 'horizontal':
        return grid[row, column:column+4]
    
    elif search_type == 'vertical':
        return grid[row:row+4, column]
    
    elif search_type == 'diag_left':
        return grid[range(row, row + 4), range(column, column + 4)]
    
    elif search_type == 'diag_right':
        return grid[range(row, row - 4, -1), range(column, column + 4)]

def game_won(grid, piece, config):

    for direction in ['horizontal', 'vertical', 'diag_left', 'diag_right']:

        # Look for all possible ways to obtain 4 in a row
        if direction == 'horizontal':
            row_range = range(config.rows)
            column_range = range(config.columns - 3)

        elif direction == 'vertical':
            row_range = range(config.rows - 3)
            column_range = range(config.columns)

        elif direction == 'diag_left':
            row_range = range(config.rows - 3)
            column_range = range(config.columns - 3)
        
        elif direction == 'diag_right':
            row_range = range(3, config.rows)
            column_range = range(config.columns - 3)
        
        # Search through all possible sequences, and see if there is a connect 4
        for row in row_range:
            for col in column_range:
                search_range = get_search_range(direction, grid, row, col)

                # If a connect 4 is found
                if (np.count_nonzero(search_range == piece) == 4 
                    and np.count_nonzero(search_range == 0) == 0):
                    return True

    return False