import os
import random
import math

def create_folder_if_not_exists(folder_name):
  """Creates a folder if it does not exist already."""
  if not os.path.exists(folder_name):
    os.mkdir(folder_name)

def print_list_with_underscores(list_to_print):
  """Prints a list with underscores between the elements."""
  string_to_print = ""
  for element in list_to_print:
    string_to_print += str(element) + "_"
  string_to_print = string_to_print[:-1]
  return string_to_print

def random_list(n, a, b):
    """
    Creates a list of `n` random integers with bounds `[a,b]`.

    Args:
        n: The number of elements in the list.
        a: The lower bound of the integers.
        b: The upper bound of the integers.

    Returns:
        A list of `n` random integers.
    """

    random_list = []
    for _ in range(n):
        random_list.append(random.randint(a, b))
    return random_list

def average_and_standard_deviation(list_of_elements):
    """
    Computes the average and standard deviation of a list of elements.

    Args:
        list_of_elements: The list of elements to compute the average and standard deviation of.

    Returns:
        A tuple of (average, standard_deviation).
    """

    # Calculate the sum of the elements.
    sum_of_elements = 0
    for element in list_of_elements:
        sum_of_elements += element

    # Calculate the number of elements.
    n = len(list_of_elements)

    # Calculate the average.
    average = sum_of_elements / n

    # Calculate the squared deviations from the mean.
    squared_deviations_from_the_mean = []
    for element in list_of_elements:
        squared_deviation_from_the_mean = (element - average) ** 2
        squared_deviations_from_the_mean.append(squared_deviation_from_the_mean)

    # Calculate the variance.
    variance = sum(squared_deviations_from_the_mean) / n

    # Calculate the standard deviation.
    standard_deviation = math.sqrt(variance)

    return average, standard_deviation

def write_string_to_file(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
    except IOError as e:
        print(f"Error writing to file: {e}")