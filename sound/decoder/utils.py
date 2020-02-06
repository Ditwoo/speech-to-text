from numba import jit


@jit(parallel=False, fastmath=True)
def levenshtein_distance(first_string: str,
                         second_string: str):
    if first_string is None:
        raise TypeError('`first_string` should not be None')
    if second_string is None:
        raise TypeError('`second_string` should not be None')

    if first_string == second_string:
        return 0

    if len(first_string) < len(second_string):
        return levenshtein_distance(second_string, first_string)

    if len(second_string) == 0:
        return len(first_string)

    insert_cost: int = 1
    delete_cost: int = 1
    swap_cost: int = 1

    previous_row = list(range(len(second_string) + 1))
    for i, c1 in enumerate(first_string):
        current_row = [i + 1]
        for j, c2 in enumerate(second_string):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + insert_cost
            deletions = current_row[j] + delete_cost  # than second_string
            substitutions = previous_row[j] + int(c1 != c2) * swap_cost
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
