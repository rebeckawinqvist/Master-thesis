import numpy as np
import scipy as sp
from scipy import sparse
from numpy import savetxt, loadtxt


# Example A
exA_A = np.array([[1, 1], [0, 1]])
exA_B = np.array([[0], [1]])
exA_xub = np.array([[5], [5]])
exA_xlb = np.array([[-5], [-5]])
exA_uub = np.array([1])
exA_ulb = np.array([-1])

exA_N = np.array([3])
exA_Q = np.eye(2)
exA_P = exA_Q
exA_R = np.array([10])

savetxt('exA_A.csv', exA_A, delimiter = ',')
savetxt('exA_B.csv', exA_B, delimiter = ',')
savetxt('exA_xub.csv', exA_xub, delimiter = ',')
savetxt('exA_xlb.csv', exA_xlb, delimiter = ',')
savetxt('exA_uub.csv', exA_uub, delimiter = ',')
savetxt('exA_ulb.csv', exA_ulb, delimiter = ',')
savetxt('exA_N.csv', exA_N, delimiter = ',')
savetxt('exA_Q.csv', exA_Q, delimiter = ',')
savetxt('exA_P.csv', exA_P, delimiter = ',')
savetxt('exA_R.csv', exA_R, delimiter = ',')


# Example B
exB_A = np.array([[1, 1], [0, 1]])
exB_B = np.array([[0], [1]])
exB_xub = np.array([[10], [10]])
exB_xlb = np.array([[-10], [-10]])
exB_uub = np.array([5])
exB_ulb = np.array([-5])

exB_N = np.array([3])
exB_Q = np.eye(2)
exB_P = exB_Q
exB_R = np.array([10])

savetxt('exB_A.csv', exB_A, delimiter = ',')
savetxt('exB_B.csv', exB_B, delimiter = ',')
savetxt('exB_xub.csv', exB_xub, delimiter = ',')
savetxt('exB_xlb.csv', exB_xlb, delimiter = ',')
savetxt('exB_uub.csv', exB_uub, delimiter = ',')
savetxt('exB_ulb.csv', exB_ulb, delimiter = ',')
savetxt('exB_N.csv', exB_N, delimiter = ',')
savetxt('exB_Q.csv', exB_Q, delimiter = ',')
savetxt('exB_P.csv', exB_P, delimiter = ',')
savetxt('exB_R.csv', exB_R, delimiter = ',')


# Example C - x unbounded
exC_A = np.array([[1, 1], [0, 1]])
exC_B = np.array([[0], [1]])
exC_uub = np.array([1])
exC_ulb = np.array([-1])

exC_N = np.array([3])
exC_Q = np.eye(2)
exC_P = exC_Q
exC_R = np.array([10])


# Example D
exD_A = np.array([[1, 1], [0, 1]])
exD_B = np.array([[0], [1]])
exD_xub = np.array([[5], [5]])
exD_xlb = np.array([[-5], [-5]])
exD_uub = np.array([0.5])
exD_ulb = np.array([-0.5])

exD_N = np.array([3])
exD_Q = np.eye(2)
exD_P = exD_Q
exD_R = np.array([10])

savetxt('exD_A.csv', exD_A, delimiter = ',')
savetxt('exD_B.csv', exD_B, delimiter = ',')
savetxt('exD_xub.csv', exD_xub, delimiter = ',')
savetxt('exD_xlb.csv', exD_xlb, delimiter = ',')
savetxt('exD_uub.csv', exD_uub, delimiter = ',')
savetxt('exD_ulb.csv', exD_ulb, delimiter = ',')
savetxt('exD_N.csv', exD_N, delimiter = ',')
savetxt('exD_Q.csv', exD_Q, delimiter = ',')
savetxt('exD_P.csv', exD_P, delimiter = ',')
savetxt('exD_R.csv', exD_R, delimiter = ',')



# Example E
exE_A = np.array([ [0.7, -0.1, 0,   0 ],
                   [0.2, -0.5, 0.1, 0 ],
                   [0,    0.1, 0.1, 0 ],
                   [0.5,  0,   0.5, 0.5]
                    ])
 
exE_A = np.array([[ 0.7,  -0.1,  0,    0 ], 
                  [ 0.2,  -0.5,  0.1,  0  ],
                  [ 0,    0.1,   0.1,  0  ],
                  [ 0.5,  0,     0.5,  0.5 ]
                  ])
exE_B = np.array([[ 0,    0.1], 
                  [ 0.1,  1. ],
                  [ 0.1,  0. ],
                  [ 0,    0  ]
                   ])

exE_xlb = np.array([[-6], [-6], [-1], [-0.5]])
exE_xub = np.array([[6], [6], [1], [0.5]])
exE_ulb = np.array([[-5], [-5]])
exE_uub = np.array([[5], [5]])

exE_N = np.array([10])
exE_Q = np.eye(4)
exE_P = exE_Q
exE_R = np.eye(2)

savetxt('exE_A.csv', exE_A, delimiter = ',')
savetxt('exE_B.csv', exE_B, delimiter = ',')
savetxt('exE_xub.csv', exE_xub, delimiter = ',')
savetxt('exE_xlb.csv', exE_xlb, delimiter = ',')
savetxt('exE_uub.csv', exE_uub, delimiter = ',')
savetxt('exE_ulb.csv', exE_ulb, delimiter = ',')
savetxt('exE_N.csv', exE_N, delimiter = ',')
savetxt('exE_Q.csv', exE_Q, delimiter = ',')
savetxt('exE_P.csv', exE_P, delimiter = ',')
savetxt('exE_R.csv', exE_R, delimiter = ',')



