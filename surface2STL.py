import numpy as np


def local_write_facet(fid, p1, p2, p3):

    if np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(p3).any():
        num = 0
    else:
        num = 1
        v1 = p2 - p1
        v2 = p3 - p1
        v3 = np.cross(v1, v2)
        n = v3 / np.sqrt(np.sum(v3 * v3))

        fid.write('{:f} {:f} {:f}'.format(n))
        fid.write('{:f} {:f} {:f}'.format(p1))
        fid.write('{:f} {:f} {:f}'.format(p2))
        fid.write('{:f} {:f} {:f}'.format(p3))

        # fid.write(str(list(map('{:f}'.format,n))))
        # fid.write(str(list(map('{:f}'.format,p1))))
        # fid.write(str(list(map('{:f}'.format,p2))))
        # fid.write(str(list(map('{:f}'.format,p3))))
        fid.write('{:01d}'.format(0))

    return num


def surface2STL(filename, x, y, z):

    nFacets = 0
    mode = 'binary'
    fid = open(filename, 'w+')
    for i in range(0, z.shape[0] - 1):
        for j in range(0, z.shape[1] - 1):

            p1 = np.array([x[i, j], y[i, j], z[i, j]])
            p2 = np.array([x[i, j+1], y[i, j+1], z[i, j+1]])
            p3 = np.array([x[i+1, j+1], y[i+1, j+1], z[i+1, j+1]])
            val = local_write_facet(fid, p1, p2, p3)
            nFacets = nFacets + val

    fid.seek(0, 0)
    fid.seek(80, 0)
    fid.write('{:01d}'.format(nFacets))
    fid.close()
