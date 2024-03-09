def knapsack_for_video_summary(values, weights, capacity, scale_factor=5):
    """
    Apply the 0/1 Knapsack algorithm to select video segments for summarization.

    :param values: List of importance scores for each segment.
    :param weights: List of durations for each segment in seconds.
    :param capacity: Maximum total duration for the summary in seconds.
    :param scale_factor: Factor to scale weights to integers.
    :return: Indices of the segments to include in the summary.
    """
    # Scale weights and capacity
    weights = [int(w * scale_factor) for w in weights]
    capacity = int(capacity * scale_factor)

    n = len(values)
    K = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # Build table K[][] in a bottom-up manner
    for i in range(n + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif weights[i-1] <= w:
                K[i][w] = max(values[i-1] + K[i-1][w-weights[i-1]], K[i-1][w])
            else:
                K[i][w] = K[i-1][w]

    # Find the selected segments
    res = K[n][capacity]
    w = capacity
    selected_indices = []

    for i in range(n, 0, -1):
        if res <= 0:
            break
        if res == K[i-1][w]:
            continue
        else:
            selected_indices.append(i-1)
            res = res - values[i-1]
            w = w - weights[i-1]

    selected_indices.reverse()
    return selected_indices
