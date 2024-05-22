from serialization import load_model
import pickle
import numpy as np

def makeObj(path=''):
    ## Load SMPL model (here we load the female model)
    ## Make sure path is correct
    m = load_model('../../models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')

    if path!='':
        m3 = pickle.load(open(path, 'rb'),encoding='iso-8859-1')
        m.pose[:] = m3['pose']
        m.betas[:] = m3['betas']
    else:
        # Assign random pose and shape parameters
        m.pose[:] = np.random.rand(m.pose.size) * .2
        m.betas[:] = np.random.rand(m.betas.size) * .03

    ## Write to an .obj file
    outmesh_path = './hello_smpl.obj'
    with open(outmesh_path, 'w') as fp:
        for v in m.r:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in m.f + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    ## Print message
    print('..Output mesh saved to: ', outmesh_path)


makeObj()