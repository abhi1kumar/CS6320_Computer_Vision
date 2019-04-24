import numpy as np
import math

def Register3DPointsQuaternion(pointsA, pointsB):
# compute transformation from pointsA and poitnsB so that
# pointsB = R * pointsA + t
# pointsA, pointsB - 3 x n matrices.
#
    numPoints = pointsA.shape[1]

    # compute centroid
    centroidA = np.mean(pointsA, 1)
    centroidB = np.mean(pointsB, 1)

    # find rotation
    pA = pointsA - centroidA[:,None];
    pB = pointsB - centroidB[:, None];

    M = np.zeros((3,3))
    for i in range(numPoints):
    	M += np.matmul(pA[:,i, None], pB[None, :,i])

    N = np.array([
	[M[0,0]+M[1,1]+M[2,2], M[1,2]-M[2,1], M[2,0]-M[0,2], M[0,1]-M[1,0]],
	[M[1,2]-M[2,1], M[0,0]-M[1,1]-M[2,2], M[0,1]+M[1,0], M[2,0]+M[0,2]],
	[M[2,0]-M[0,2], M[0,1]+M[1,0], -M[0,0]+M[1,1]-M[2,2], M[1,2]+M[2,1]],
	[M[0,1]-M[1,0], M[2,0]+M[0,2], M[1,2]+M[2,1], -M[0,0]-M[1,1]+M[2,2]]
    ])

    # w: eigenvalues
    # v: normalized eigenvectors
    w, v = np.linalg.eig(N)
    qmin = v[:, np.argmax(w)]

    R = Quaternion2R(qmin)
    # find translation given rotation
    rotPointsA = np.matmul(R, pointsA);
    rotCentroidA = np.mean(rotPointsA, 1);
    t = centroidB - rotCentroidA;
    finalTrans = RT2Trans(R,t)

    return finalTrans

def RT2Trans(R,t):
    trans = np.eye(4)
    trans[0:3,0:3] = R
    trans[0:3,3] = t
    return trans;

def Quaternion2R(q):
    q = q / np.linalg.norm(q)

    sq1, sq2, sq3, sq4 = np.multiply(q, q)
    q12, q13, q14, q23, q24, q34 = q[0]*q[1], q[0]*q[2], q[0]*q[3], q[1]*q[2], q[1]*q[3], q[2]*q[3]

    R = [[sq1 + sq2 - sq3 - sq4,  2*(q23 - q14),  2*(q24+q13)],
	[2*(q23 + q14),  sq1 - sq2 + sq3 - sq4,   2*(q34 - q12)],
	[2*(q24 - q13),  2*(q34 + q12),  sq1 - sq2 - sq3 + sq4]]
    return R;

def rotationVectorToMatrix(rvec):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    Please refer to https://www.wikiwand.com/en/Euler%E2%80%93Rodrigues_formula
    """
    rvec = np.asarray(rvec)
    theta = math.sqrt(np.dot(rvec, rvec))
    axis = rvec/theta;
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc - ad), 2 * (bd + ac)],
                     [2 * (bc + ad), aa + cc - bb - dd, 2 * (cd - ab)],
                     [2 * (bd - ac), 2 * (cd + ab), aa + dd - bb - cc]])

if __name__ == '__main__':
    pointsA = np.transpose(np.array([[5,6,8], [10, 2, 3], [18, 9, 10]]));
    trueRotMat = rotationVectorToMatrix([0.4,-0.2,0.3]);
    truePose = np.transpose(np.array([10,7,33]))
    trueTrans = RT2Trans(trueRotMat, truePose)

    pointsB = np.matmul(trueTrans, np.vstack((pointsA, np.ones(3))))
    pointsB = np.divide(pointsB, pointsB[3,:])
    pointsB = pointsB[0:3,:]
    print(trueTrans)

    finalTrans = Register3DPointsQuaternion(pointsA, pointsB)
    print(finalTrans)
