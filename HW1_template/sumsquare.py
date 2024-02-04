import numpy.testing as npt
import numpy as np
import timeit
import pandas as pd
import matplotlib.pyplot as plt


def gen_random_samples(n):
    """
    Generate n random samples using the
    numpy random.randn module.

    Returns
    ----------
    sample : 1d array of size n
        An array of n random samples
    """
    ## TODO FILL IN
    return np.random.randn(n)


def sum_squares_for(samples):
    """
    Compute the sum of squares using a forloop

    Parameters
    ----------
    samples : 1d-array with shape n
        An array of numbers.

    Returns
    -------
    ss : float
        The sum of squares of the samples
    """
    total = 0
    for sample in samples:
        total += sample**2
    return total


def sum_squares_np(samples):
    """
    Compute the sum of squares using Numpy's dot module

    Parameters
    ----------
    samples : 1d-array with shape n
        An array of numbers.

    Returns
    -------
    ss : float
        The sum of squares of the samples
    """
    return np.dot(samples, samples)


def time_ss(sample_list):
    """
    Time it takes to compute the sum of squares
    for varying number of samples. The function should
    generate a random sample of length s (where s is an
    element in sample_list), and then time the same random
    sample using the for and numpy loops.

    Parameters
    ----------
    samples : list of length n
        A list of integers to .

    Returns
    -------
    ss_dict : Python dictionary with 3 keys: n, ssfor, ssnp.
        The value for each key should be a list, where the
        ordering of the list follows the sample_list order
        and the timing in seconds associated with that
        number of samples.
    """
    ss_dict = {"n": [], "ssfor": [], "ssnp": []}

    for n in sample_list:
        # random samples of size n
        samples = np.random.randn(n)

        start = timeit.default_timer()
        sum_squares_for(samples)
        elapsed_for = timeit.default_timer() - start
        ss_dict["ssfor"].append(elapsed_for)

        start = timeit.default_timer()
        sum_squares_np(samples)
        elapsed_np = timeit.default_timer() - start
        ss_dict["ssnp"].append(elapsed_np)

        ss_dict["n"].append(n)

    return ss_dict


def timess_to_df(ss_dict):
    """
    Time the time it takes to compute the sum of squares
    for varying number of samples.

    Parameters
    ----------
    ss_dict : Python dictionary with 3 keys: n, ssfor, ssnp.
        The value for each key should be a list, where the
        ordering of the list follows the sample_list order
        and the timing in seconds associated with that
        number of samples.

    Returns
    -------
    time_df : Pandas dataframe that has n rows and 3 columns.
        The column names must be n, ssfor, ssnp and follow that order.
        ssfor and ssnp should contain the time in seconds.
    """
    time_df = pd.DataFrame(ss_dict, columns=["n", "ssfor", "ssnp"])
    return time_df


def main():
    # generate 10 samples
    samples = gen_random_samples(100)
    # call the for version
    ss_for = sum_squares_for(samples)
    # call the numpy version
    ss_np = sum_squares_np(samples)
    # make sure they are approximately the same value
    npt.assert_almost_equal(ss_for, ss_np, decimal=5)

    n_values = [10, 100, 1000, 10000, 100000, 1000000, 10000000]

    timing_data = time_ss(n_values)

    timing_df = timess_to_df(timing_data)

    plt.figure(figsize=(10, 6))
    plt.plot(
        timing_df["n"],
        timing_df["ssfor"],
        "o-",
        label="For Loop",
        color="red",
    )
    plt.plot(
        timing_df["n"],
        timing_df["ssnp"],
        "s-",
        label="Numpy",
        color="blue",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Samples (n)")
    plt.ylabel("Time Taken (seconds)")
    plt.title("Comparison of Sum of Squares Calculation Time Across 10 to 10 Million")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
