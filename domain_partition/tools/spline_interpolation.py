def get_streamlines_as_splines(streamlines=None):

    splines = []

    for i in range(len(streamlines)):
        streamline = np.array(streamlines[i])
        x = streamline[:, 0]
        y = streamline[:, 1]
        if x.size == 2:
            tck, u = splprep([x, y], s=0, k=1)  # k=1 linear splines
            splines.append([tck, u])
        else:
            tck, u = splprep([x, y], s=0)  # Cubic Splines (Default)
            splines.append([tck, u])

    return splines


