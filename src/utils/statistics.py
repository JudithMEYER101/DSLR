def ft_count(data):
    """Return lenght of data"""

    return len(data)


def ft_mean(data):
    """Return mean"""

    if not data:
        return None
    return sum(data) / len(data)


def ft_min(data):
    """Return minimum"""

    if not data:
        return None
    minimum = data[0]
    for value in data:
        if value < minimum:
            minimum = value
    return minimum


def ft_max(data):
    """Return maximum"""

    if not data:
        return None
    maximum = data[0]
    for value in data:
        if value > maximum:
            maximum = value
    return maximum


def ft_percentile(data, percent):
    """Calculate percentile"""

    if not data:
        return None

    n = len(data)
    pos = percent * (n - 1)
    lower = int(pos)
    upper = lower + 1

    if upper >= n:
        return data[lower]

    fraction = pos - lower
    return data[lower] + (data[upper] - data[lower]) * fraction


def ft_std(data):
    """Calculate standard deviation"""

    if len(data) < 2:
        return None
    mean = ft_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return variance ** 0.5